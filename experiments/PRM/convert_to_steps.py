# This file takes a math_datasetN.csv file, and for every question/answer pair it parses it into steps.

from tqdm import tqdm
import json
import argparse
import datasets

def split_rollout(rollout):
    # First, only take whats after Response:\n
    #rollout = rollout.split("ASSISTANT:\n")
    steps = rollout.split("Step")
    steps = [step for step in steps if step != ""]
    # remove the " N)" from each step
    steps = [step[3:] for step in steps]
    # remove any empty steps
    steps = [step for step in steps if step != ""]

    return steps

def parse_args():
    # takes a json file as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="The input json file")
    parser.add_argument("--output_file", type=str, required=False, help="The output file")
    args = parser.parse_args()
    return args

def create_dict(dataset):
    # creates a dict from the dataset
    return {
        "question": list(map(lambda x: x[0], dataset)),
        "answer" : list(map(lambda x: x[3], dataset)),
        "dialogue": list(map(lambda x: x[1], dataset)),
        "steps": list(map(lambda x: x[4], dataset)),

    }

if __name__ == "__main__":
    args = parse_args()

    # load JSON
    with open(args.input_file) as f:
        data = json.load(f)

    # use tqdm, map through split_rollout
    for idx, elem in tqdm(enumerate(data['data'])):
        data['data'][idx].append(split_rollout(elem[1]))
    
    # create a HF dataset
    dataset = datasets.Dataset.from_dict(create_dict(data['data']))
    # save to output file, if not specified, save to input file with _steps appended
    if args.output_file == None:
        args.output_file = args.input_file[:-5] + "_steps/"

    dataset.save_to_disk(args.output_file)
