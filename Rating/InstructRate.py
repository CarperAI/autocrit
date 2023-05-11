from string import Template
import torch
from datasets import load_dataset, Dataset
import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F


@torch.no_grad()
def evaluate_acc(model, tokenizer, train, test, A_token, B_token):
    results = list()
    with tqdm.tqdm(train) as t:
        fp = 0
        fn = 0
        tp = 0
        ratio = []
        for item in t:
            rev_pos_tokens = tokenizer(item[2], return_tensors='pt')
            rev_pos_tokens = rev_pos_tokens['input_ids']
            rev_pos_tokens = rev_pos_tokens.to(model.device)
            if rev_pos_tokens.shape[1] > 2048:
                continue
            pos = torch.log(F.softmax(model(tokenizer(item[0], return_tensors='pt')['input_ids'].to(model.device)).logits, dim=-1)[:, -1])
            neg = torch.log(F.softmax(model(tokenizer(item[1], return_tensors='pt')['input_ids'].to(model.device)).logits, dim=-1)[:, -1])
            rev_pos = torch.log(F.softmax(model(rev_pos_tokens).logits, dim=-1)[:, -1])
            rev_neg = torch.log(F.softmax(model(tokenizer(item[3], return_tensors='pt')['input_ids'].to(model.device)).logits, dim=-1)[:, -1])
            pos_score = pos[A_token] - pos[B_token]
            neg_score = neg[B_token] - neg[A_token]
            rev_pos_score = rev_pos[B_token] - rev_pos[A_token]
            rev_neg_score = rev_neg[A_token] - rev_neg[B_token]
            tot_score = (pos_score + neg_score + rev_neg_score + rev_pos_score).item()
            if item[4]:
                ratio.append(1)
                if tot_score >= 0:
                    tp += 1
                else:
                    fn += 1
            else:
                ratio.append(0)
                if tot_score >= 0:
                    fp += 1
                else:
                    tp += 1
            t.set_description(f"Acc: {tp/(fp+tp+fn):.3f}, "
                              f"rP:{np.mean(np.array(ratio)):.3f}, "
                              f"FP: {fp}, "
                              f"FN: {fn}, "
                              f"last: pos {pos_score.item():.3f}, neg {neg_score.item():.3f}, "
                              f"rev_pos {rev_pos_score.item():.3f}, rev_neg {rev_neg_score.item():.3f}")



def preprocess_dataset(
    dataset: Dataset,
    template: Template,
    reverse_template: Template,
    option_a,
    option_b,
    initial_instruct: str,
    reversed_initial_instruct: str,
):
    train_data = list()
    test_data = list()
    for item in dataset['train']:
        train_data.append([
            template.substitute(
                passage=item['prompt'],
                option_a=option_a,
                option_b=option_b,
                initial_instruct_passage=initial_instruct
            ),
            template.substitute(
                passage=item['prompt'],
                option_a=option_b,
                option_b=option_a,
                initial_instruct_passage=reversed_initial_instruct
            ),
            reverse_template.substitute(
                passage=item['prompt'],
                option_a=option_a,
                option_b=option_b,
                initial_instruct_passage=initial_instruct
            ),
            reverse_template.substitute(
                passage=item['prompt'],
                option_a=option_b,
                option_b=option_a,
                initial_instruct_passage=reversed_initial_instruct
            ),
            item['helpful']
        ])
    for item in dataset['test']:
        test_data.append([
            template.substitute(
                passage=item['prompt'],
                option_a=option_a,
                option_b=option_b,
                initial_instruct_passage=initial_instruct
            ),
            template.substitute(
                passage=item['prompt'],
                option_a=option_b,
                option_b=option_a,
                initial_instruct_passage=reversed_initial_instruct
            ),
            reverse_template.substitute(
                passage=item['prompt'],
                option_a=option_a,
                option_b=option_b,
                initial_instruct_passage=initial_instruct
            ),
            reverse_template.substitute(
                passage=item['prompt'],
                option_a=option_b,
                option_b=option_a,
                initial_instruct_passage=reversed_initial_instruct
            ),
            item['helpful']
        ])
    return train_data, test_data


if __name__ == "__main__":
    template = Template(
        """### Human: $passage

Question: Please select between one of the below two choices based on if the critique is helpful.
Choice A: $option_a
Choice B: $option_b
### Assistant: $initial_instruct_passage
"""
    )
    reverse_template = Template(
        """### Human: $passage

Question: Please select between one of the below two choices based on if the critique is unhelpful.
Choice A: $option_a
Choice B: $option_b
### Assistant: $initial_instruct_passage
"""
    )

    option_a = "true"
    option_b = "false"
    initial_instruct = f"Between choice A: {option_a} and choice B: {option_b} if I had to pick one it is choice"
    reversed_initial_instruct = f"Between choice A: {option_b} and choice B: {option_a} if I had to pick one it is choice"
    model = AutoModelForCausalLM.from_pretrained(
        "models/stable-vicuna-13b-fp16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).cuda()
    A_token = 319
    B_token = 350
    tokenizer = AutoTokenizer.from_pretrained("models/stable-vicuna-13b-fp16")
    train, test = preprocess_dataset(
        load_dataset("dmayhem93/self-critiquing-helpful-rate"), template, reverse_template, option_a, option_b, initial_instruct, reversed_initial_instruct
    )
    evaluate_acc(model, tokenizer, train, test, A_token, B_token)
