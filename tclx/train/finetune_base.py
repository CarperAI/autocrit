import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import TrainerCallback, AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy
import json
from torch.nn.utils.rnn import pad_sequence
import argparse
import wandb
from datasets import load_dataset
import random

from tclx.utils.utils import load_yaml, load_jsonl, freeze_bottom_causal_layers
from tclx.data.datasets import SFTDataset, MaskedSFTDataset


def call_model(model, prompts, batch_size=16, max_length=500):
    answers = []
    tok = model.tok
    prompts = [tok(prompt, return_tensors="pt").input_ids.flatten() for prompt in prompts]
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    prompt_batches = [prompts[i * batch_size : (i+1) * batch_size] for i in range(num_batches)]
    for i, batch in enumerate(prompt_batches):
        batch = [torch.flip(prompt, dims=[0]) for prompt in batch]
        batch = pad_sequence(batch, batch_first=True, padding_value=tok(tok.pad_token).input_ids[0])
        batch = torch.flip(batch, dims=[1])
        batch = batch.cuda()
        output = model.generate(batch, max_length=max_length, do_sample=True)
        output = tok.batch_decode(output)
        answers += output
    return answers


class SampleCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        epoch = state.epoch
        dataset = eval_dataloader.dataset
        prompts = dataset.prompts[:16]
        responses = call_model(model, prompts)
        if torch.distributed.get_rank() == 0:
            response_table = wandb.Table(columns=["prompts", "responses"], data=[[prompt, response] for prompt, response in zip(prompts, responses)])
            wandb.log({"generations": response_table})


def train(config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    tokenizer.pad_token = tokenizer.eos_token
    config.update({"eval_accumulation_steps": 2, "fp16_full_eval": True})
    training_args = TrainingArguments(**config["train_args"])
    model = AutoModelForCausalLM.from_pretrained(config["model_path"])
    model.tok = tokenizer
    model.cuda()

    data = load_dataset(config["data_path"])["train"]
    val_size = int(len(data) * 0.02)
    val_data = data.select([i for i in range(val_size)])
    data = data.select([val_size + i for i in range(len(data) - val_size)])
    
    print("Len data: ", len(data))

    if config["trainer"] == "unmasked":
        train_dataset = SFTDataset(data, tokenizer)
        val_dataset = SFTDataset(val_data, tokenizer)
    elif config["trainer"] == "masked":
        train_dataset = MaskedSFTDataset(data, tokenizer)
        val_dataset = MaskedSFTDataset(val_data, tokenizer)
    else:
        raise ValueError("{} is unsupported train type!".format(config["trainer"]))

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, callbacks=[SampleCallback],
            eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                'attention_mask': torch.stack([f[1] for f in data]),
                                                                'labels': torch.stack([f[2] for f in data])})
    trainer.train()
    model.save_pretrained(config["train_args"]["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--ds_config_path", type=str)
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    config = load_yaml(args.config_path)
    config["train_args"]["deepspeed"] = args.ds_config_path

    train(config)