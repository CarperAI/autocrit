import os
import random
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from autocrit.configs import ModelConfig


def set_seed(seed=None, deterministic=False) -> int:
    if seed is None:
        seed = np.random.default_rng().integers(2**32 - 1)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # torch.use_deterministic_algorithms(deterministic)
    return seed


def truncate_code(completion: str, def_num=1, print_num=0, only_local_scope=False):
    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in ["^#", re.escape("<|endoftext|>"), "^'''", '^"""', "\n\n\n"]
    ]
    if print_num > 0:
        prints = list(re.finditer("^print", completion, re.MULTILINE))
        if print_num >= 0 and len(prints) > print_num:
            completion = completion[: prints[print_num].start()]

    if only_local_scope:
        global_lines = list(re.finditer("^[a-zA-Z]", completion, re.MULTILINE))
        if global_lines:
            completion = completion[: global_lines[0].start()]
    else:
        defs = list(re.finditer("^def", completion, re.MULTILINE))
        if len(defs) > def_num:
            completion = completion[: defs[def_num].start()]

    start_pos = 0

    terminals_pos = [
        pos
        for pos in [find_re(completion, terminal, start_pos) for terminal in terminals]
        if pos != -1
    ]
    if len(terminals_pos) > 0:
        return completion[: min(terminals_pos)]
    else:
        return completion


def model_setup(cfg: ModelConfig, device=None):
    set_seed(cfg.seed)

    if device is None:
        device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.model_max_length > 32768:
        tokenizer.model_max_length = 2048

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if cfg.fp16 else None,
        low_cpu_mem_usage=cfg.fp16,
    ).to(device)

    return model, tokenizer, device
