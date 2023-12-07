import argparse
import os
import sys
from typing import List, Optional, Dict, Any, Union

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from huggingface_hub import list_repo_refs
from matplotlib import pyplot
from rich.console import Console
from rich.table import Table
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          LlamaConfig, LlamaModel, PreTrainedModel)


# for any model which doesn't have a ForSequenceClassification wrapper in transformers
class ClassificationModel(PreTrainedModel):
    def __init__(self, model_path):
        super().__init__(AutoConfig.from_pretrained(model_path, trust_remote_code=True))
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).model
        self.score = torch.nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask, **kwargs):
        logits = self.score(self.llm(input_ids, attention_mask=attention_mask, **kwargs)[0])
        sequence_lengths = (torch.eq(input_ids, self.llm.config.pad_token_id).long().argmax(-1) - 1).to(logits.device)
        pooled_logits = logits[torch.arange(input_ids.shape[0], device=logits.device), sequence_lengths]
        return pooled_logits[None]

# wrapper for UltraRM model
class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward( # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        transformer_outputs = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,
                            )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)

        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)
        rewards = torch.gather(rewards, 1, ends)

        return rewards[None]

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="reciprocate/tiny-mistral", type=str)
parser.add_argument("--revision", default=None, type=str)
parser.add_argument("--tokenizer_path", default=None, type=str)
parser.add_argument("--dataset", default="reciprocate/number-pairs", type=str)
parser.add_argument("--lr", default=6e-3, type=float)
parser.add_argument("--min_lr", default=None, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--epochs", default=4, type=int)
parser.add_argument("--seq_length", default=2048, type=int)
parser.add_argument("--num_unfrozen_layers", default=None, type=int)
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--load_in_4bit", action="store_true")
parser.add_argument("--downscale_weight", action="store_true")
parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
parser.add_argument("--eval_interval", default=100, type=int)
parser.add_argument("--only_eval", action="store_true")
parser.add_argument("--wrapper", action="store_true")
parser.add_argument("--wrapper_ultra", action="store_true")
parser.add_argument("--format", default="alpaca", type=str)
parser.add_argument("--calibration_datasets", default=[], nargs="+", type=str)
args = parser.parse_args(args=[] if "__file__" not in globals() else sys.argv[1:])

def plot_calibration(model_name: str, dataset_name: str, delta_scores: np.ndarray) -> str:
    space = np.linspace(0, 4, 32)
    perfect_calibration = 1 / (1 + np.exp(-space))

    epsilon = 1 / 4
    probs = []
    for center in space:
        ixs = (center - epsilon < abs(delta_scores)) & (abs(delta_scores) < center + epsilon)
        if not ixs.any():
            prob = 0.5
        else:
            prob = np.mean(delta_scores[ixs] > 0)

        probs.append(prob)

    textcolor = "#333"
    matplotlib.style.use("ggplot")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 15,
        "text.color": textcolor,
        "axes.labelcolor": textcolor,
        "axes.labelpad": 12,
        "xtick.color": textcolor,
        "ytick.color": textcolor,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "figure.titlesize": 14,
        "figure.figsize": (12, 8),
    })
    pyplot.plot(space, perfect_calibration, label="perfect calibration", c="grey")
    pyplot.plot(space, probs, label=model_name)

    ax = pyplot.gca()
    ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=True, left=False, labelleft=True)
    ax.set_facecolor("#fff")
    ax.set_title(f"Calibration on {dataset_name}", size=26, y=1.02, fontdict={"fontweight": "normal"})
    ax.set_xlabel("Score difference", size=26)
    ax.set_ylabel("Accuracy", size=26)
    pyplot.legend(loc="best", fontsize=20, title_fontproperties={"weight": "normal", "style": "normal"}, fancybox=False, frameon=False)
    pyplot.tight_layout()

    os.makedirs("calibrations", exist_ok=True)
    image_path = os.path.join("calibrations", f"{model_name}@{dataset_name}.png".replace("/", "_"))
    pyplot.savefig(image_path, dpi=64)
    pyplot.clf()
    return image_path


if __name__ == "__main__":
    seed = int(os.environ.get("RANK", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if args.revision is None:
        if os.path.exists(args.model_path):
            revision = "local"
        else:
            revision = list_repo_refs(args.model_path).branches[0].target_commit[:8]
    model_name = f"{args.model_path.split('/')[-1]}"

    experiment = os.environ.get("EXPERIMENT")
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name="autocrit",
        config=vars(args),
        init_kwargs={"wandb": {"name": experiment if experiment else f"{model_name}@{args.dataset}"}}
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    def format(sample: Dict[str, Union[list, str]], tokenizer: AutoTokenizer) -> Dict[str, str]:
        """
        There are multiple formats the datasets can be in, this function converts formats them
        into a single string pair of (selected, rejected):

        1. sample = {selected: str, rejected: str}
        assume the samples are already formatted

        2. sample = {prompt: str, selected: str, rejected: str}
        prepend the prompt to the selected and rejected and format them as a single turn chatml

        3. sample = {selected: chatml, rejected: chatml}
        format the selected and rejected according to the tokenizer's template
        """

        if isinstance(sample["selected"], str) and isinstance(sample["rejected"], str):
            if "prompt" not in sample:
                return {
                    "selected": sample["selected"],
                    "rejected": sample["rejected"],
                }

            selected = [{"role": "assistant", "content": sample["selected"]}]
            rejected = [{"role": "assistant", "content": sample["rejected"]}]

            if isinstance(sample["prompt"], str):
                prompt = [{"role": "user", "content": sample["prompt"]}]
            
            selected = prompt + selected
            rejected = prompt + rejected

        else:
            selected = sample["selected"]
            rejected = sample["rejected"]

        return {
            "selected": tokenizer.apply_chat_template(selected, tokenize=False),
            "rejected": tokenizer.apply_chat_template(rejected, tokenize=False),
        }

    def tokenize(x: Dict[str, str], tokenizer: AutoTokenizer) -> Dict[str, torch.LongTensor]:
        return {
            "selected_input_ids": tokenizer(x["selected"], truncation=True, max_length=args.seq_length).input_ids,
            "rejected_input_ids": tokenizer(x["rejected"], truncation=True, max_length=args.seq_length).input_ids,
        }

    def collate_fn(batch):
        input_ids = sum([[x["rejected_input_ids"], x["selected_input_ids"]] for x in batch], [])
        return tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")

    dataset_name, split = args.dataset.split(":") if ":" in args.dataset else (args.dataset, None)
    if os.path.exists(dataset_name):
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name)

    if "test" in dataset:
        args.calibration_datasets.append(f'{args.dataset}:test')
        ref_dataset_name = f'{args.dataset}:test'
    else:
        if len(args.calibration_datasets):
            ref_dataset_name = args.calibration_datasets[0]
        else:
            ref_dataset_name = args.dataset

    dataset_names = [args.dataset, *args.calibration_datasets]
    dataset_name, *eval_dataset_names = dataset_names
    datasets = []

    for name in dataset_names:
        if ':' in name:
            dataset_path, split = name.split(':')
        else:
            dataset_path = name
            split = None

        if os.path.exists(dataset_path):
            dataset = load_from_disk(dataset_path)
        else:
            dataset = load_dataset(dataset_path)

        if split is not None:
            dataset = dataset[split]
        else:
            if name in eval_dataset_names and "test" in dataset:
                dataset = dataset["test"]
            elif "train" in dataset:
                dataset = dataset["train"]
            else:
                raise ValueError(f"There is no 'train' or 'test' split in `{name}`")

        if "chosen" in dataset.column_names:
            dataset = dataset.rename_column("chosen", "selected")
        if "question" in dataset.column_names:
            dataset = dataset.rename_column("question", "prompt")
        if "replies" in dataset.column_names:
            dataset = dataset.map(lambda x: {"selected": x["replies"][0], "rejected": x["replies"][1]}, remove_columns=["replies"])

        dataset = dataset.map(format, fn_kwargs=dict(tokenizer=tokenizer), desc="Formatting")

        if accelerator.is_main_process:
            table = Table(title=f"Dataset: {name}", show_lines=True)
            table.add_column("selected")
            table.add_column("rejected")

            # replace forward slash with the next closest unicode character just for printing
            # because rich treats it as a formatting character
            clean_tags = lambda s: s.replace("[/INST]", "[∕INST]")

            for i in range(3):
                table.add_row(clean_tags(dataset[i]["selected"]), clean_tags(dataset[i]["rejected"]))

            console = Console()
            console.print(table)

        datasets.append(dataset)

    tokenized_datasets = []
    for dataset in datasets:
        tokenized_datasets.append(dataset.map(tokenize, fn_kwargs=dict(tokenizer=tokenizer), desc="Tokenizing"))

    dataset, *eval_datasets = tokenized_datasets

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)
    eval_dataloaders = []
    for eval_dataset in eval_datasets:
        eval_dataloaders.append(torch.utils.data.DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn))

    if args.wrapper:
        model = ClassificationModel.from_pretrained(args.model_path)
        model.llm.resize_token_embeddings(len(tokenizer))
        model.llm.config.pad_token_id = tokenizer.pad_token_id
    elif args.wrapper_ultra:
        model = LlamaRewardModel.from_pretrained(args.model_path, torch_dtype=torch.float16)
        model.model.resize_token_embeddings(len(tokenizer))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path, revision=args.revision, num_labels=1, trust_remote_code=True, load_in_4bit=args.load_in_4bit)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        if isinstance(model, ClassificationModel):
            model.llm.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()

    if args.downscale_weight:
        model.score.weight.data *= 0.1

    if args.num_unfrozen_layers is not None and args.num_unfrozen_layers > 0:
        frozen = False

        try:
            for layer in model.transformer.h[:-args.num_unfrozen_layers]:
                layer.requires_grad_(False)
            frozen = True
        except AttributeError:
            pass

        try:
            for layer in model.model.layers[:-args.num_unfrozen_layers]:
                layer.requires_grad_(False)
            frozen = True
        except AttributeError:
            pass

        if not frozen:
            raise ValueError("Could not freeze layers, modify the code to support your architecture.")

    if args.only_eval:
        model, *eval_dataloaders = accelerator.prepare(model, *eval_dataloaders)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=len(dataloader) * args.epochs, eta_min=args.min_lr or args.lr)
        model, opt, scheduler, dataloader, *eval_dataloaders, = accelerator.prepare(model, opt, scheduler, dataloader, *eval_dataloaders)

    best_accuracy = 0
    step = 0

    tbar = tqdm(range(args.epochs * len(dataloader)), disable=not accelerator.is_main_process or args.only_eval)
    for iepoch in range(args.epochs):
        for batch in dataloader:
            if step % args.eval_interval == 0 or step == tbar.total - 1:
                accuracies = {}
                statistics = {}
                for dataset_name, eval_dataloader in zip(eval_dataset_names, eval_dataloaders):
                    model.eval()
                    all_scores, all_delta_scores, all_tokens = [], [], []
                    main_scores, main_tokens = [], []

                    for batch in tqdm(eval_dataloader, desc=f"Evaluating on {dataset_name}", disable=not accelerator.is_main_process, leave=args.only_eval):
                        with torch.no_grad():
                            scores = model(**batch)[0]
                            delta_scores = scores.reshape(-1, 2).diff().view(-1)

                        main_scores.extend(scores.view(-1).tolist())
                        main_tokens.extend(batch["input_ids"].tolist())

                        scores = accelerator.gather_for_metrics(scores.view(-1))
                        delta_scores = accelerator.gather_for_metrics(delta_scores)

                        all_scores.extend(scores.tolist())
                        all_delta_scores.extend(delta_scores.tolist())

                    delta_scores = np.hstack(all_delta_scores)
                    scores = np.hstack(all_scores)
                    accuracy = (delta_scores > 0).mean()
                    accuracies[dataset_name] = accuracy

                    if accelerator.is_main_process:
                        image_path = plot_calibration(model_name, dataset_name, delta_scores)

                        texts = [text.replace(tokenizer.pad_token, "") for text in tokenizer.batch_decode(main_tokens)]
                        samples = wandb.Table(["text", "score"], rows=list(zip(texts, main_scores))[:128])

                        postfix = f"@{dataset_name.split('/')[-1]}"
                        stats = {
                            f"delta_scores{postfix}": delta_scores,
                            f"delta_scores/mean{postfix}": delta_scores.mean(),
                            f"delta_scores/std{postfix}": delta_scores.std(),
                            f"delta_scores/min{postfix}": delta_scores.min(),
                            f"delta_scores/max{postfix}": delta_scores.max(),
                            f"scores{postfix}": wandb.Histogram(scores),
                            f"scores/mean{postfix}": scores.mean(),
                            f"scores/std{postfix}": scores.std(),
                            f"scores/min{postfix}": scores.min(),
                            f"scores/max{postfix}": scores.max(),
                        }

                        statistics[dataset_name] = stats

                        accelerator.log({
                            f"accuracy{postfix}": accuracy,
                            f"calibration{postfix}": wandb.Image(image_path),
                            f"samples{postfix}": samples,
                            **stats,
                        }, step=step)

                        tbar.set_postfix(accuracy=accuracies.get(ref_dataset_name), best_accuracy=best_accuracy)

                if accuracies.get(ref_dataset_name, 0) > best_accuracy:
                    best_accuracy = accuracies.get(ref_dataset_name, 0)

                    best_accuracies = {f"accuracy@{k}@best": v for k, v in accuracies.items()}
                    best_statistics = [{f"{d}/{k}@best": v for k, v in xs.items()} for d, xs in statistics.items()]
                    best_statistics = {k: v for xs in best_statistics for k, v in xs.items()}

                    accelerator.log({
                        "best_accuracy": best_accuracy,
                        **best_accuracies,
                        **best_statistics,
                    }, step=step)

                    if not args.only_eval:
                        path = f"{model_name}_{args.dataset}_{experiment}_{args.lr}".replace("/", "_").replace(":", "_").replace("@", "_")
                        accelerator.unwrap_model(model).save_pretrained(
                            os.path.join(args.checkpoint_dir, path),
                            save_function=accelerator.save,
                            is_main_process=accelerator.is_main_process,
                            state_dict=accelerator.get_state_dict(model),
                        )
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(os.path.join(args.checkpoint_dir, path))
                        accelerator.print(f"Checkpoint: {os.path.join(args.checkpoint_dir, path)}")

                    tbar.set_postfix(accuracy=accuracies.get(ref_dataset_name), best_accuracy=best_accuracy)

                accelerator.wait_for_everyone()
                if args.only_eval:
                    exit()

                model.train()

            with accelerator.accumulate(model):
                scores = model(batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=not args.gradient_checkpointing)[0]

                if "scores" in batch:
                    loss = F.mse_loss(scores.view(-1), batch["scores"].view(-1))
                else:
                    delta_scores = scores.reshape(-1, 2).diff()
                    loss = -F.logsigmoid(delta_scores).mean()

                accelerator.backward(loss)
                opt.step()
                opt.zero_grad()
                scheduler.step()

            tbar.update()
            tbar.set_description(f"Training {args.model_path} on {args.dataset}; loss: {loss.item():.4f}")
            accelerator.log({"loss": loss.item(), "lr": float(scheduler.get_last_lr()[0])}, step=step)
            step += 1
