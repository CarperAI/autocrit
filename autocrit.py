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
def generate(model, tokenizer, prompts, temperature=1, max_new_tokens=256, max_length=2048, stop=[]):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
    all_ids = model.generate(**inputs, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=True, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    output_ids = all_ids[:, inputs.input_ids.shape[1]:]
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    for i in range(len(outputs)):
        for s in stop:
            if s in outputs[i]:
                outputs[i] = outputs[i][:outputs[i].index(s)]

    return outputs

def generate_openai(prompt, model="gpt-3.5-turbo", max_new_tokens=128, system_prompt="", temperature=1, stop=[]):
    MAX_API_RETRY = 5
    for _ in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
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

def revise(prompts, get_answer, get_critique, constitution=[], max_iterations=3, score_fn=None):
    accelerator = Accelerator()
    dataloader = accelerator.prepare(DataLoader(prompts))
    outputs = []

    stop = ["\nREVISION REQUEST:", "\nCRITIQUE REQUEST:", "\nUSER:", "\nASSISTANT:", "\nREVISION:", "\nCRITIQUE:"]
    get_answer = truncate_output(get_answer, stop)
    get_critique = truncate_output(get_critique, stop)

    for (question,) in tqdm(dataloader, disable=not accelerator.is_main_process):
        iterations = []
        context = f"USER: {question}"
        print0(context)

        if constitution:
            constitution_iter = islice(cycle(map(lambda x: x.values(), constitution)), max_iterations)

        for i in range(max_iterations):
            role = "ASSISTANT" if i == 0 else "REVISION"
            context += f"\n{role}: "
            answer = get_answer(context)
            context += answer
            print0(f"{role}: {answer}", style="bold")

            if constitution:
                critique_request, revision_request = next(constitution_iter)
                context += f"\nCRITIQUE REQUEST: {critique_request}"
                print0(f"CRITIQUE REQUEST: {critique_request}")

            context += "\nCRITIQUE: "
            critique = get_critique(context)
            context += critique
            print0(f"CRITIQUE: {critique}", style="italic")

            if constitution:
                context += f"\nREVISION REQUEST: {revision_request}"
                print0(f"REVISION REQUEST: {revision_request}")

            if score_fn:
                score = score_fn(question=question, answer=answer)
                print0(f"SCORE: {score}", style="blue")
            else:
                score = None

            iterations.append((answer, critique, score, context))

        outputs.append({
            "question": question,
            "iterations": [{"answer": answer, "critique": critique, "score": score, "context": context} for answer, critique, score, context in iterations]
        })

    return gather(outputs, total_size=len(prompts))

def gather(samples, total_size: int):
    if not torch.distributed.is_initialized():
        return samples

    all_samples = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(all_samples, samples)
    return sum(all_samples, [])[:total_size]

def truncate_output(generate_fn, stop=[]):
    def fn(*args, **kwargs):
        output = generate_fn(*args, **kwargs)

        for s in stop:
            if s in output:
                output = output[:output.index(s)]
        return output
    return fn

def finetune(accelerator, model, tokenizer, optim, prompts, outputs, eval_prompts=[]):
    samples = []
    for prompt, output in zip(prompts, outputs):
        tokenized = tokenizer([prompt.strip(), output.strip()])
        labels = tokenized.input_ids.copy()
        labels[0] = [-100] * len(labels[0])
        samples.append({
            "input_ids": sum(tokenized.input_ids, []),
            "attention_mask": sum(tokenized.attention_mask, []),
            "labels": sum(labels, []),
        })

    dataloader = accelerator.prepare(DataLoader(samples, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer)))

    epochs = 4
    total_steps = epochs * len(dataloader)
    tbar = trange(total_steps, disable=not accelerator.is_main_process)
    for epoch in range(epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                optim.step()
                optim.zero_grad()

            loss_global = accelerator.reduce(loss, "mean")
            tbar.set_description(f"loss: {loss_global.item():g}")
            tbar.update()

