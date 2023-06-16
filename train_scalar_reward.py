import torch
import wandb
import argparse
import os
import numpy as np
from tqdm import tqdm
from time import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from huggingface_hub import list_repo_refs
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from accelerate import Accelerator
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="reciprocate/gpt2-tiny", type=str)
parser.add_argument("--revision", default=None, type=str)
parser.add_argument("--tokenizer_path", default=None, type=str)
parser.add_argument("--dataset", default="reciprocate/number-pairs", type=str)
parser.add_argument("--lr", default=6e-4, type=float)
parser.add_argument("--min_lr", default=None, type=float)
parser.add_argument("--weight_decay", default=0.1, type=float)
parser.add_argument("--batch_size", default=20, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--seq_length", default=1024, type=int)
parser.add_argument("--num_unfrozen_layers", default=None, type=int)
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--load_in_4bit", action="store_true")
parser.add_argument("--downscale_weight", action="store_true")
parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
parser.add_argument("--eval_interval", default=100, type=int)
parser.add_argument("--only_eval", action="store_true")
parser.add_argument("--add_oasst_tokens", action="store_true")
parser.add_argument("--calibration_datasets", default=[], nargs="+", type=str)
args = parser.parse_args()


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

    import matplotlib
    from matplotlib import pyplot

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
    model_name = f"{args.model_path}:{revision}"

    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name="autocrit",
        config={"batch_size": args.batch_size, "lr": args.lr},
        init_kwargs={"wandb": {"name": f"{model_name}@{args.dataset}"}},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)

    if args.add_oasst_tokens:
        tokenizer.add_tokens(["<|assistant|>", "<|prefix_begin|>", "<|prefix_end|>", "<|prompter|>", "<|system|>"])
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    def to_vicuna_format(sample):
        return {
            "prompt": sample["prompt"].replace("\n\nHuman:", "USER:").replace("\n\nAssistant:", " ASSISTANT:")[4:],
        }

    def tokenize(prompt, selected, rejected, tokenizer):
        return {
            "selected_input_ids": tokenizer(prompt + selected + tokenizer.eos_token, truncation=True, max_length=args.seq_length).input_ids,
            "rejected_input_ids": tokenizer(prompt + rejected + tokenizer.eos_token, truncation=True, max_length=args.seq_length).input_ids,
        }

    def collate_fn(batch):
        input_ids = sum([[x["rejected_input_ids"], x["selected_input_ids"]] for x in batch], [])
        return tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")

    dataset = load_dataset(args.dataset)
    if "chosen" in dataset["train"].column_names:
        dataset = dataset.rename_column("chosen", "selected")
    if "replies" in dataset["train"].column_names:
        dataset = dataset.map(lambda x: {"selected": x["replies"][0], "rejected": x["replies"][1]}, remove_columns=["replies"])
    accelerator.print(args.dataset, dataset)

    eval_dataloaders = []
    for name in args.calibration_datasets:
        calibration_dataset = load_dataset(name)
        if "test" in calibration_dataset:
            calibration_dataset = calibration_dataset["test"]
        else:
            calibration_dataset = calibration_dataset["train"]

        accelerator.print(name, calibration_dataset)
        tokenized = calibration_dataset.map(tokenize, input_columns=["prompt", "selected", "rejected"], fn_kwargs=dict(tokenizer=tokenizer), desc="Tokenizing")
        dataloader = torch.utils.data.DataLoader(tokenized, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn)
        eval_dataloaders.append(dataloader)

    tokenized = dataset.map(tokenize, input_columns=["prompt", "selected", "rejected"], fn_kwargs=dict(tokenizer=tokenizer), desc="Tokenizing")
    dataloader = torch.utils.data.DataLoader(tokenized["train"], shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)
    eval_dataloaders.append(torch.utils.data.DataLoader(tokenized["test"], shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn))

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, revision=args.revision, num_labels=1, load_in_4bit=args.load_in_4bit)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
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
                for dataset_name, eval_dataloader in zip(args.calibration_datasets + [args.dataset], eval_dataloaders):
                    model.eval()
                    all_scores, all_delta_scores, all_tokens = [], [], []

                    for batch in tqdm(eval_dataloader, desc=f"Evaluating on {dataset_name}", disable=not accelerator.is_main_process, leave=args.only_eval):
                        with torch.no_grad():
                            scores = model(**batch)[0]

                        delta_scores = scores.reshape(-1, 2).diff().view(-1)
                        delta_scores = accelerator.gather_for_metrics(delta_scores)
                        all_delta_scores.extend(delta_scores.tolist())
                        all_scores.extend(scores.view(-1).tolist())
                        all_tokens.extend(batch["input_ids"].tolist())

                    delta_scores = np.hstack(all_delta_scores)
                    accuracy = (delta_scores > 0).mean()

                    if accelerator.is_main_process:
                        image_path = plot_calibration(model_name, dataset_name, delta_scores)

                        texts = [text.replace(tokenizer.pad_token, "") for text in tokenizer.batch_decode(all_tokens)]
                        samples = wandb.Table(["text", "score"], rows=list(zip(texts, all_scores))[:128])

                        postfix = "" if dataset_name == args.dataset else f"@{dataset_name.split('/')[-1]}"
                        accelerator.log({
                            f"accuracy{postfix}": accuracy,
                            f"samples{postfix}": samples,
                            f"delta_scores{postfix}": delta_scores,
                            f"calibration{postfix}": wandb.Image(image_path),
                        }, step=step)

                    if accuracy > best_accuracy and dataset_name == args.dataset:
                        best_accuracy = accuracy
                        accelerator.log({"best_accuracy": best_accuracy}, step=step)

                        if args.only_eval:
                            exit()
                        else:
                            path = f"{model_name}_{args.dataset}".replace("/", "_").replace(":", "_").replace("@", "_")
                            accelerator.unwrap_model(model).save_pretrained(
                                os.path.join(args.checkpoint_dir, path),
                                save_function=accelerator.save,
                                is_main_process=accelerator.is_main_process,
                                state_dict=accelerator.get_state_dict(model),
                            )
                            if accelerator.is_main_process:
                                tokenizer.save_pretrained(os.path.join(args.checkpoint_dir, path))
                            accelerator.print(f" Checkpointing -> {os.path.join(args.checkpoint_dir, path)}")

                    if dataset_name == args.dataset:
                        tbar.set_postfix(accuracy=accuracy, best_accuracy=best_accuracy)

                accelerator.wait_for_everyone()
                model.train()

            with accelerator.accumulate(model):
                scores = model(**batch, use_cache=not args.gradient_checkpointing)[0]
                loss = -F.logsigmoid(scores.reshape(-1, 2).diff()).mean()
                accelerator.backward(loss)
                opt.step()
                opt.zero_grad()
                scheduler.step()

            tbar.update()
            tbar.set_description(f"Training {args.model_path} on {args.dataset}; loss: {loss.item():.4f}")
            accelerator.log({"loss": loss.item(), "lr": float(scheduler.get_last_lr()[0])}, step=step)
            step += 1
