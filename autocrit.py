import os
import random
import time

import numpy as np
import openai
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from itertools import cycle, islice

from rich.console import Console

console = Console(width=80)
print = console.print
print0 = lambda *args, **kwargs: print(*args, **kwargs) if os.environ.get("RANK", "0") == "0" else None

@torch.inference_mode()
def generate(model, tokenizer, user_prompts, temperature=1, max_new_tokens=256, max_length=4096, stop=[]):
    inputs = tokenizer(user_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
    all_ids = model.generate(**inputs, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=True, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    output_ids = all_ids[:, inputs.input_ids.shape[1]:]
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    for i in range(len(outputs)):
        for stop_sequence in stop:
            if stop_sequence in outputs[i]:
                outputs[i] = outputs[i][:outputs[i].index(stop_sequence)]

    return outputs

def generate_openai(user_prompt, max_new_tokens=128, system_prompt="", temperature=1, stop=[]):
    MAX_API_RETRY = 5
    for _ in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_new_tokens,
                stop=stop,
            )

            return response["choices"][0]["message"]["content"]

        except Exception as e:
            print(e)
            time.sleep(10)

    raise Exception(f"Failed after {MAX_API_RETRY} retries.")

def critique_loop(prompts, get_answer, get_critique, constitution=[], max_iter=3):
    dataloader = Accelerator().prepare(DataLoader(prompts))
    outputs = []

    for (prompt,) in tqdm(dataloader, disable=not Accelerator().is_main_process):
        context = f"USER: {prompt}"
        print0(f"USER: {prompt}")
        iterations = []

        if constitution:
            constitution_iter = islice(cycle(map(lambda x: x.values(), constitution)), max_iter)

        for i in range(max_iter):
            role = "ASSISTANT" if i == 0 else "REVISION"
            context += f"\n{role}:"
            answer = get_answer(context)
            context += answer
            print0(f"{role}: {answer}", style="bold")

            if constitution:
                critique_request, revision_request = next(constitution_iter)
                context += f"\nCRITIQUE REQUEST: {critique_request}"
                print0(f"CRITIQUE REQUEST: {critique_request}")

            context += "\nCRITIQUE:"
            critique = get_critique(context)
            context += critique
            print0(f"CRITIQUE: {critique}", style="italic")

            if constitution:
                context += f"\nREVISION REQUEST: {revision_request}"
                print0(f"REVISION REQUEST: {revision_request}")

            iterations.append((context, answer, critique))

        outputs.append({
            "prompt": prompt,
            "iterations": [{"context": context, "answer": answer, "critique": critique} for context, answer, critique in iterations]
        })

    return gather(outputs, total_size=len(prompts))

def gather(samples, total_size: int):
    if not torch.distributed.is_initialized():
        return samples

    all_samples = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(all_samples, samples)
    return sum(all_samples, [])[:total_size]

def finetune(model, tokenizer, prompts, outputs, eval_prompts=[]):
    accelerator = Accelerator()

    samples = []
    for prompt, output in zip(prompts, outputs):
        tokenized = tokenizer([prompt, output])
        labels = tokenized.input_ids.copy()
        labels[0] = [-100] * len(labels[0])
        samples.append({
            "input_ids": sum(tokenized.input_ids, []),
            "attention_mask": sum(tokenized.attention_mask, []),
            "labels": sum(labels, []),
        })

    dataloader = DataLoader(samples, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))
    eval_dataloader = DataLoader(eval_prompts, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.99))
    model, opt, dataloader, eval_dataloader = accelerator.prepare(model, opt, dataloader, eval_dataloader)

    i = 0
    epochs = 1
    total_steps = epochs * len(dataloader)
    eval_every = total_steps
    tbar = trange(total_steps, disable=not accelerator.is_main_process)
    for epoch in range(epochs):
        for batch in dataloader:
            if i % eval_every == 0 or i == total_steps - 1:
                model.eval()
                outputs = []
                for eval_prompts in tqdm(eval_dataloader, disable=not accelerator.is_main_process):
                    outputs.extend(generate(accelerator.unwrap_model(model), tokenizer, eval_prompts))

                outputs = gather(outputs, len(eval_prompts))

                for output in outputs:
                    print(f"{'*' * 80}{output}")
                model.train()

            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                opt.step()
                opt.zero_grad()

            loss_global = accelerator.reduce(loss, "mean")
            tbar.set_description(f"loss: {loss_global.item():g}")
            i += 1
            tbar.update()

