from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from torch.utils.data import Dataset, random_split
from tqdm import tqdm
from datasets import load_dataset
from logger import Logger
from accelerate import Accelerator
import argparse
from transformers import StoppingCriteria, StoppingCriteriaList
import os
from utils import make_rm, load_rm
import random


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = []):
      StoppingCriteria.__init__(self), 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
      self.stops = stops
      for i in range(len(stops)):
        self.stops = self.stops[i]

stopping_criteria = None


def infer_clm(model, dataloader, max_length, temp):
    """Function to infer causal model in parallel on dataloader with at most max_length tokens at temperature temp.
    """
    for inputs in tqdm(dataloader):
        tok_prompts = inputs[0]
        data = inputs[1]
        with torch.no_grad():
            if os.environ.get('DEEPSPEED_ZERO_STAGE', '0') != '3':
                # Note: synced_gpus=True only needed for zero3
                outputs = model.generate(**tok_prompts, max_length=max_length, do_sample=True, temperature=temp, stopping_criteria=stopping_criteria)[:, tok_prompts["input_ids"].shape[1]:]
            else:
                outputs = model.generate(**tok_prompts, max_length=max_length, do_sample=True, temperature=temp, stopping_criteria=stopping_criteria, synced_gpus=True)[:, tok_prompts["input_ids"].shape[1]:]
        text_outputs = tokenizer.batch_decode(outputs)
        for sample, output in zip(data, text_outputs):
            sample["response"] = output
        # Log responses
        Logger.log(data)
        return data


def infer_rm(model, dataloader):
    """Function to infer reward model in parallel on dataloader
    """
    for inputs in tqdm(dataloader):
        tok_prompts = inputs[0]
        data = inputs[1]
        with torch.no_grad():
            outputs = model(**tok_prompts)
            outputs = outputs.tolist()
        for sample, output in zip(data, outputs):
            sample["reward"] = output[0]
        # Log rewards
        Logger.log(data)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dataset", type=str)
    parser.add_argument("--log_file", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--rm_path", default="")
    parser.add_argument("--order", nargs="*", default=["response"])
    parser.add_argument("--save_model", default=False, action="store_true", help="Boolean indicating whether the reward model contains a reference to its underlying causal model")
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--temp", default=0.7, type=float)
    args = parser.parse_args()

    assert args.log_file is not None
    assert args.save_model is not None
    Logger.init(args.log_file)

    dataset = load_dataset(args.prompt_dataset)[args.split]
    dataset = [{key: sample[key] for key in sample} for sample in dataset]
    #prompt_dataset = prompt_dataset[:128]

    if args.rm_path != "":
        model = load_rm(args.model_name, args.tokenizer_name, args.rm_path, args.save_model)
        prompt_dataset = []
        for m in args.order:
            prompt_dataset += [{"prompt": sample["prompt"] + sample[m] + "<|endoftext|>", "type": m, "id": i} for i, sample in enumerate(dataset)]
        #random.shuffle(prompt_dataset)
    else:
        prompt_dataset = dataset
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()
    model.half()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Add padding on right if RM, otherwise on left
    tokenizer.padding_side = "left" if args.rm_path == "" else "right"
    tokenizer.truncation_side = "left"

    # Stop generation on next Human utterance
    stop_words_ids = [tokenizer.encode("Human:")]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    max_length = args.max_length
    def data_collator(data):
        prompts = [sample["prompt"] for sample in data]
        tok_prompts = tokenizer(prompts, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
        tok_prompts["input_ids"] = tok_prompts["input_ids"].to("cuda")
        tok_prompts["attention_mask"] = tok_prompts["attention_mask"].to("cuda")
        return tok_prompts, data

    batch_size = args.batch_size
    temp = args.temp

    dataloader = torch.utils.data.DataLoader(prompt_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    model = accelerator.unwrap_model(model)
    model = model.to(accelerator.device)

    if args.rm_path != "":
        infer_rm(model, dataloader, max_length, temp)
    else:
        infer_clm(model, dataloader, max_length, temp)