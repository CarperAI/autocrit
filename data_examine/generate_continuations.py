import datasets
import argparse
import torch
import tqdm
from transformers import (
    pipeline,
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
import json


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do some continuations")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="dmayhem93/self-critiquing-base")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    generator = pipeline(
        "text-generation",
        model=AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ),
        tokenizer=tokenizer,
        device=args.device,
    )
    data = datasets.load_dataset(args.dataset)
    set_seed(42 + args.device)
    outs = list()
    stop = StopOnTokens([tokenizer.eos_token_id, tokenizer.bos_token_id])
    for i in tqdm.tqdm(range(len(data["test"]))):
        text = data["test"][i]["prompt"]
        if len(tokenizer(text)["input_ids"]) > 1436:
            continue
        output_text = generator(
            text,
            max_length=2048,
            num_return_sequences=1,
            stopping_criteria=StoppingCriteriaList([stop]),
        )
        outs.append(
            {
                "prompt": text,
                "continuation": output_text[0]["generated_text"].replace(text, ""),
                "real": data["test"][i]["response"],
            }
        )
    with open(
        f'continuations_{args.dataset.replace("/", "_")}_{args.model.replace("/", "_")}.json',
        "w",
    ) as f:
        f.write(json.dumps(outs))
