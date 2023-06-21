import argparse
import os
import torch
import transformers
from typing import Tuple, Any, Optional
import tritonclient.grpc.aio as grpcclient
from utils import triton_call

'''
We're using inference hooks rather than directly using hugging face generate incase we want to switch to a triton client at some point.
This gives us significantly improved flexibility, as autocrit is not built around a single inference API
'''


# Inference hook that takes a model and a batch of inputs and returns a batch of outputs
class InferenceHook:
    def __init__(self, dir : str):
        self.dir = dir
        self.API_KEY = ""
        self.API_URL = ""
    
    def load(self, **kwargs):
        pass

    # Calls the inference API and returns the result
    def infer(self, *args: Any, **kwds: Any) -> Any:
        pass




# Inference hook that uses the HuggingFace API to call a model
class HuggingFaceHook(InferenceHook):
    def __init__(self, dir : str, tokenizer_name : Optional[str] = None):
        super().__init__(dir)
        self.model_name = dir
        if tokenizer_name is None:
            self.tokenizer_name = dir
        else:
            self.tokenizer_name = tokenizer_name

    def load(self, **kwargs):
        # Load the model and tokenizer
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)

    def infer(self, *args: Any, **kwds: Any) -> Any:
        # model.generate, use the tokenizer to convert the output to text and kwds for gen arguments
        # assume input and output are batched
        inp_text = kwds["input_texts"]
        inp_ids = self.tokenizer(inp_text, return_tensors="pt", padding=True).input_ids
        out_ids = self.model.generate(inp_ids, **kwds)
        out_text = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return out_text
    
class TritonHook(InferenceHook):
    def __init__(self, dir : str, model_name : str, tokenizer_name : Optional[str] = None):
        super().__init__(dir)
        self.url = dir # url contains host:port
        self.model_name = model_name
        if tokenizer_name is None:
            self.tokenizer_name = model_name
        else:   
            self.tokenizer_name = tokenizer_name

        self.client = None
        self.tokenizer = None

    def load(self, **kwargs):
        # create a client using url
        self.client = grpcclient.InferenceServerClient(url=self.url)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)


    def infer(self, *args: Any, **kwds: Any) -> Any:
        # use self.client to call the model
        # assume input and output are batched
        inp_text = kwds["input_texts"]
        inp_ids = self.tokenizer(inp_text, return_tensors="pt", padding=True).input_ids
        # call infer
        logits, output_ids = triton_call(self.client, self.model_name, inp_ids, **kwds)
        out_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return out_text
    
