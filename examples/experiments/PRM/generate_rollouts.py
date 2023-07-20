from autocrit.inference.inference_hook import HuggingFaceHook, HuggingFaceHookBestOfN, TextGenerationHook

# This file uses WizardLM to generate an initial set of CoT rollouts using the MATH dataset
# as a prompt. The rollouts are then used to train a PRM model, after being fed to a critique model.   

import argparse
import os
import torch
import transformers
import datasets
from typing import Tuple, Any
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed


'''
We're using inference hooks rather than directly using hugging face generate incase we want to switch to a triton client at some point.
This gives us significantly improved flexibility, as autocrit is not built around a single inference API
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="ehartford/WizardLM-13B-Uncensored", help="Path to HF checkpoint with the base model"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()


    # load the MATH dataset, using datasets. Only use the first 1% of train
    dataset = datasets.load_dataset("math_dataset", "algebra__linear_1d", split="train[:10%]")

    # Load the model and tokenizer
    inference_hook = TextGenerationHook("http://26.0.146.24:8080")
    inference_hook.load()


    # write a prompt that solicits the answer using CoT
    def construct_prompt_wizard7b(input_text):
        prompt = "Instruction: " + input_text + "Think step by step about how you would solve this problem. Number your steps.\n\n\
Response:\n Step 1)"
        return prompt
    def construct_prompt_wizard13b(input_text):
        formula = input_text + "Think step by step about how you would solve this problem. Number your steps."
        prompt="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. 
USER: """+formula+"""
ASSISTANT:
Step 1)"""
        return prompt

    # write an example algebra problem
    input_text = "Solve 2x + 3 = 7 for x.\n\n"
    prompt = construct_prompt_wizard13b(input_text)

    # inference
    generate_params = {
        "do_sample": True,
        "top_k": 10,
        "max_new_tokens": 300,
        "best_of": 2,
    }
    rollouts = inference_hook.infer(input_texts=prompt, generate_params=generate_params)

    # every step is seperated by "Step n)", where n is the step number
    # split the rollouts into steps
    def split_rollout(rollout):
        # First, only take whats after Response:\n
        rollout = rollout.split("ASSISTANT:\n")

        steps = rollout.split("Step")
        steps = [step for step in steps if step != ""]
        # remove the N) from each step
        steps = [step[3:] for step in steps]
        # remove any empty steps
        steps = [step for step in steps if step != ""]

        return steps
    # create a dataloader for the dataset. Only use the first 10% of the dataset for now
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    """
    # create a list that is [question, complete prompt+response, steps, answer] and save it to a file
    # batch over the dataset to save memory
    output_dataset = []
    last_save_idx=0

    for question in tqdm(train_dataloader): 
        input_text = question["question"]

        # map the input text to a prompt
        prompts = list(map(construct_prompt_wizard13b, input_text))
        # inference
        rollouts = inference_hook.infer(input_texts=prompts, generate_params=generate_params)
        # if rollouts is a string and not a list, make it a list
        if type(rollouts) == str:
            rollouts = [rollouts]

        # split the rollouts into steps
        steps = list(map(split_rollout, rollouts))

        for idx in range(len(steps)):
            output_dataset.append([input_text[idx], rollouts[idx], steps[idx], question["answer"][idx]])
        
        # every 100 questions, save the dataset
        if len(output_dataset) - last_save_idx >= 100:
            df = pd.DataFrame(output_dataset, columns=["question", "prompt", "steps", "answer"])
            df.to_csv("math_dataset"+str(last_save_idx)+".csv", index=False)
            last_save_idx = len(output_dataset)
    
    """
    
    def execute_and_wrap(lam, kwargs):
        # execute lam, and return (lam(), kwargs)
        return lam(), kwargs

    # We're going to rewrite the above using ThreadPoolExecutor to speed up inference. 
    executor = ThreadPoolExecutor(max_workers=16)
    futures = []
    for idx, qa in tqdm(enumerate(train_dataloader)):
        input_text = qa["question"]
        # map the input text to a prompt
        prompts = list(map(construct_prompt_wizard13b, input_text))

        # inference
        inference_lambda = lambda: inference_hook.infer(input_texts=prompts, generate_params=generate_params)
        kwargs = {
            "input_text": input_text,
            "qa": qa,
        }
        futures.append(executor.submit(execute_and_wrap, inference_lambda, kwargs))

    # Every time another 500 finish, save the dataset. Use as_completed to get the results
    last_save_idx = 0
    output_dataset = []
    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            rollouts, kwargs = future.result()

            input_text = kwargs["input_text"]
            qa = kwargs["qa"]

            # if rollouts is a string and not a list, make it a list
            output_dataset.append([input_text[0], rollouts, qa["question"][0], qa["answer"][0]])
            
            # every 100 questions, save the dataset
            if len(output_dataset) - last_save_idx >= 500:
                df = pd.DataFrame(output_dataset, columns=["prompt", "rollout", "question", "answer"])
                df.to_json("math_dataset"+str(last_save_idx)+".json", orient="split")
                last_save_idx = len(output_dataset)
        except:
            print("Error in future")
            continue
    # save the dataset using pandas
    df = pd.DataFrame(output_dataset, columns=["prompt", "rollout", "question", "answer"])
    df.to_json("math_dataset.json", index=False)
