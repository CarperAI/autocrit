import os
import torch
import torch.nn.functional as F
import numpy as np
from huggingface_hub import list_repo_refs
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from accelerate import Accelerator
from tqdm import tqdm
from time import time
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="reciprocate/gpt2-tiny")
parser.add_argument("--revision", type=str, default=None)
parser.add_argument("--dataset", type=str, default="reciprocate/number-pairs")
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--batch_size", default=10, type=int)
parser.add_argument("--seq_length", default=1024, type=int)
parser.add_argument("--eval_interval", default=10, type=int)
parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
parser.add_argument("--only_eval", action="store_true")
args = parser.parse_args()


def plot_calibration(model_name: str, dataset_name: str, delta_scores: np.ndarray) -> str:
    space = np.linspace(0, 4, 32)
    perfect_calibration = 1 / (1 + np.exp(-space))

    epsilon = 1 / 4
    probs = []
    for center in space:
        ixs = (center - epsilon < abs(delta_scores)) & (abs(delta_scores) < center + epsilon)
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
        "xtick.color": textcolor,
        "ytick.color": textcolor,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "figure.titlesize": 14,
        "figure.figsize": (12, 8),
    })
    pyplot.plot(space, perfect_calibration, label="Perfect calibration", c="grey")
    pyplot.plot(space, probs, label=model_name)

    ax = pyplot.gca()
    ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=True, left=False, labelleft=True)
    ax.set_facecolor("#fff")
    ax.set_title(f"Preference calibration on {dataset_name}", size=26, y=1.02, fontdict={"fontweight": "normal"})
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    dataset = load_dataset(args.dataset)
    if "chosen" in dataset["train"].column_names:
        dataset = dataset.rename_column("chosen", "selected")
    accelerator.print(dataset)

    def tokenize(prompt, chosen, rejected):
        return {
            "selected_input_ids": tokenizer(prompt + chosen + tokenizer.eos_token, truncation=True, max_length=args.seq_length).input_ids,
            "rejected_input_ids": tokenizer(prompt + rejected + tokenizer.eos_token, truncation=True, max_length=args.seq_length).input_ids,
        }

    def collate_fn(batch):
        prefered = {"input_ids": [x["selected_input_ids"] for x in batch]}
        rejected = {"input_ids": [x["rejected_input_ids"] for x in batch]}
        prefered = tokenizer.pad(prefered, padding=True, return_tensors="pt")
        rejected = tokenizer.pad(rejected, padding=True, return_tensors="pt")
        return {
            "selected_input_ids": prefered.input_ids,
            "rejected_input_ids": rejected.input_ids,
            "selected_attention_mask": prefered.attention_mask,
            "rejected_attention_mask": rejected.attention_mask,
        }

    tokenized = dataset.map(tokenize, input_columns=["prompt", "selected", "rejected"])
    dataloader = torch.utils.data.DataLoader(tokenized["train"], shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)
    eval_dataloader = torch.utils.data.DataLoader(tokenized["test"], shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, revision=args.revision)
    model.config.num_labels = 1
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.only_eval:
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        model, opt, dataloader, eval_dataloader = accelerator.prepare(model, opt, dataloader, eval_dataloader)

    best_accuracy = 0
    step = 0
    eval_step = 0
    epochs = 1
    tbar = tqdm(range(epochs * len(dataloader)), disable=not accelerator.is_main_process or args.only_eval)

    for _ in range(epochs):
        for batch in dataloader:
            if step % args.eval_interval == 0 or step == tbar.total - 1:
                start_time = time()
                ntokens = 0
                model.eval()
                all_scores, all_delta_scores, all_tokens = [], [], []

                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    with torch.no_grad():
                        selected_scores = model(batch["selected_input_ids"], attention_mask=batch["selected_attention_mask"])[0]
                        rejected_scores = model(batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"])[0]

                    delta_scores = selected_scores - rejected_scores
                    delta_scores, accelerator.gather_for_metrics(delta_scores.view(-1))

                    all_delta_scores.extend(delta_scores.tolist())
                    all_scores.extend(torch.hstack((selected_scores, rejected_scores)).view(-1).tolist())
                    all_tokens.extend(sum(map(list, zip(batch["selected_input_ids"].tolist(), batch["rejected_input_ids"].tolist())), []))

                    ntokens_batch = batch["selected_input_ids"].numel() + batch["rejected_input_ids"].numel()
                    ntokens += accelerator.reduce(torch.tensor(float(ntokens_batch), device=accelerator.device), "mean").item()

                throughput = int(ntokens / (time() - start_time))

                delta_scores = np.hstack(all_delta_scores)
                accuracy = (delta_scores > 0).mean()

                if accelerator.is_main_process:
                    image_path = plot_calibration(model_name, args.dataset, delta_scores)

                    texts = tokenizer.batch_decode(all_tokens, skip_special_tokens=True)
                    samples = wandb.Table(["text", "score"], rows=list(zip(texts, all_scores))[:128])

                    tbar.set_postfix(accuracy=accuracy, throughput=throughput)
                    accelerator.log({
                        "accuracy": accuracy,
                        "samples": samples,
                        "delta_scores": delta_scores,
                        "throughput_tokens": throughput,
                        "calibration": wandb.Image(image_path),
                    }, step=step)

                if args.only_eval:
                    exit()

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    accelerator.log({"best_accuracy": best_accuracy}, step=step)

                    path = f"{model_name}@{args.dataset}%{best_accuracy}".replace("/", "_").replace(":", "_").replace("@", "_")
                    model.save_pretrained(os.path.join(args.checkpoint_dir, path))
                    tokenizer.save_pretrained(os.path.join(args.checkpoint_dir, path))

                model.train()

            selected_scores = model(batch["selected_input_ids"], attention_mask=batch["selected_attention_mask"])[0]
            rejected_scores = model(batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"])[0]

            loss = -F.logsigmoid(selected_scores - rejected_scores).mean()
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()
            tbar.update()
            tbar.set_description(f"loss: {loss.item():.4f}")
            accelerator.log({"loss": loss.item()}, step=step)
            step += 1
