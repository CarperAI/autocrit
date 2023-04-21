import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import TrainerCallback, AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy, AutoModel, AutoConfig, PreTrainedModel, AutoModelForSequenceClassification
import json
import deepspeed
import argparse
from datasets import load_dataset
import wandb
import random

from tclx.data.datasets import RankedDataset, RankedEvalDataset, ranked_data_collator
from tclx.utils.utils import freeze_bottom_causal_layers, load_yaml, make_rm


model = None
order = None
trainer = None
NUM_SAMPLES = 16

class RankedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Batch ordered most to least preferable
        rewards = model(**inputs)
        loss = 0
        comps = rewards.view((-1, model.num_reps))
        bs, num_pairs = comps.shape[0], model.num_reps * (model.num_reps - 1) / 2
        for comp in comps:
            # Compute pairwise diffs then sum upper diagonal
            diffs = -torch.log(torch.sigmoid(comp.unsqueeze(1) - comp))
            # triu not implemented for bf16 so must initialize mask in fp
            mask = torch.triu(torch.full((model.num_reps, model.num_reps), 1, device=diffs.device), diagonal=1).to(dtype=diffs.dtype)
            loss += torch.sum(mask * diffs) / num_pairs
        loss /= bs
        return (loss, rewards) if return_outputs else loss


# Compute accuracy metric in a callback because compute_metrics not called when model is not `Pretrained` hf model
class EvaluateCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        preds = torch.tensor(trainer.predict(trainer.eval_dataset)[0]).view(-1, model.num_reps)
        bs, num_pairs = preds.shape[0], model.num_reps * (model.num_reps - 1) / 2
        inputs = model.tok.batch_decode([trainer.eval_dataset[i][0].flatten().tolist() for i in range(model.num_reps * NUM_SAMPLES)])
        num_correct = 0
        samples = {m: [] for m in order + ["scores"]}
        for i, comp in enumerate(preds):
            diffs = comp.unsqueeze(1) - comp
            # triu not implemented for bf16 so must initialize mask in fp
            mask = torch.triu(torch.full((model.num_reps, model.num_reps), 1, device=diffs.device), diagonal=1).to(dtype=diffs.dtype)
            num_correct += torch.sum(mask * diffs > 0)
            # Log samples for wandb
            if i < NUM_SAMPLES:
                for m, response in zip(order, inputs[i * model.num_reps : (i+1) * model.num_reps]):
                    samples[m].append(response)
                samples["scores"].append(comp.tolist())

        # Log samples
        if torch.distributed.get_rank() == 0:
            wandb.log({"samples": wandb.Table(data=pd.DataFrame(samples))})
            acc = num_correct / (bs * num_pairs)
            wandb.log({"accuracy": acc})


def train(config):
    global model, order, trainer
    config.update({"eval_accumulation_steps": 2, "fp16_full_eval": True})
    training_args = TrainingArguments(**config["train_args"])
    
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    tokenizer.pad_token = tokenizer.eos_token
    
    model = make_rm(config["model_path"], config["model_type"], config["tokenizer_path"], True)
    freeze_bottom_causal_layers(model, config["num_layers_unfrozen"])
    PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]
    model.PAD_ID = PAD_ID
    model.config.pad_token_id = model.config.eos_token_id
    model.tok = tokenizer

    max_length = config["max_tokens"]

    if torch.distributed.get_rank() == 0:
        wandb.init(project="tclx", config=config)

    data = load_dataset(config["data_path"])
    train_data = data["train"]
    if data.get("test") is not None:
        eval_data = data["test"]
    else:
        split = data["train"].train_test_split(test_size=0.05)
        train_data = split["train"]
        eval_data = split["test"]

    order = config["order"] if config.get("order") is not None else ["chosen", "rejected"]
    model.num_reps = len(order)
    train_dataset = RankedDataset(train_data, tokenizer, max_length=max_length, order=order, max_num=config["max_train_size"])
    eval_dataset = RankedEvalDataset(eval_data, tokenizer, max_length=max_length, order=order, max_num=config["max_train_size"])

    trainer = RankedTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=ranked_data_collator, callbacks=[EvaluateCallback])
    trainer.train()
        

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
