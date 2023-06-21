from autocrit.inference.inference_hook import HuggingFaceHook

# This file uses WizardLM to generate an initial set of CoT rollouts using the MATH dataset
# as a prompt. The rollouts are then used to train a PRM model, after being fed to a critique model.   

import argparse
import os
import torch
import transformers
from typing import Tuple, Any

'''
We're using inference hooks rather than directly using hugging face generate incase we want to switch to a triton client at some point.
This gives us significantly improved flexibility, as autocrit is not built around a single inference API
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, default="WizardLM/WizardLM-7B-V1.0", help="Path to HF checkpoint with the base model"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Load the model and tokenizer
    inference_hook = HuggingFaceHook(args.model)

    # load the MATH dataset from hugging face hub
    dataset = transformers.load_dataset("math_dataset", "algebra__linear_1d_composed", split="train")

    # write a prompt that solicits the answer using CoT
    def construct_prompt(input_text):
        prompt = "Solve for x: " + input_text + ". Think step by step about how you would solve this problem.\nAnswer:\n1."
        return prompt

    # generate the rollouts