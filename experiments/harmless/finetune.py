import os
import sys
import json
import torch
import argparse
import autocrit
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from generate import evaluate_unsafe

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revisions_path", type=str, default="artifacts/revisions-StableBeluga-7B-harmful_behaviors.json")
    parser.add_argument("--model_path", type=str, default="stabilityai/StableBeluga-7B")
    parser.add_argument("--data_path", type=str, default="https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv")
    args = parser.parse_args(args=[] if "__file__" not in globals() else sys.argv[1:])

    if args.data_path.endswith(".csv"):
        dataset = load_dataset("csv", data_files=args.data_path, split="train")
        dataset = dataset.rename_column("goal", "text")
        dataset = dataset.train_test_split(test_size=0.1, seed=0)
    else:
        dataset = load_dataset(args.data_path)

    vicuna_format = "USER: {prompt} ASSISTANT: "
    eval_prompts = dataset["test"]["text"]
    eval_prompts = [vicuna_format.format(prompt=prompt) for prompt in eval_prompts]

    with open(args.revisions_path) as f:
        revisions = json.load(f)

    prompts, outputs = [], []
    for x in revisions:
        if x["iterations"][-1]["score"] > x["iterations"][0]["score"]:
            prompts.append(x["question"])
            outputs.append(x["iterations"][-1]["answer"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    accelerator = Accelerator()
    if accelerator.state.deepspeed_plugin:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).eval()
    model.resize_token_embeddings(len(tokenizer))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95))
    model, optim = accelerator.prepare(model, optim)

    stop=["USER:", "User:", "user:", "ASSISTANT:", "Assistant:", "assistant:"]
    prior_completions = autocrit.generate(model, tokenizer=tokenizer, prompts=eval_prompts, stop=stop)
    autocrit.finetune(accelerator, model, tokenizer, optim, prompts, outputs)
    after_completions = autocrit.generate(model, tokenizer=tokenizer, prompts=eval_prompts, stop=stop)

    model.save_pretrained(f"checkpoints/finetuned-{args.model_path.split('/')[-1]}")
    if accelerator.is_main_process:
        tokenizer.save_pretrained(f"checkpoints/finetuned-{args.model_path.split('/')[-1]}")

    if accelerator.is_main_process:
        prior_scores = evaluate_unsafe(eval_prompts, prior_completions)
        after_scores = evaluate_unsafe(eval_prompts, after_completions)

        with open(f"artifacts/evaluation-{args.model_path.split('/')[-1]}.json", "w") as f:
            json.dump([{"prompt": prompt, "prior": prior, "after": after, "prior_score": prior_score, "after_score": after_score}
                       for prompt, prior, after, prior_score, after_score in zip(eval_prompts, prior_completions, after_completions, prior_scores, after_scores)], f, indent=2)



