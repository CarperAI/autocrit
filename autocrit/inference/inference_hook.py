import argparse
import os
import torch
import transformers
from typing import Tuple, Any, Optional, List, Dict
import tritonclient.grpc.aio as grpcclient
from autocrit.inference.utils import triton_call

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
    def infer(self, input_texts : List[str], 
              generate_params : Dict[str, Any], 
              **kwargs: Any) -> Any:
        """
        input_texts: a list of strings, each string is a prompt
        generate_params: a dictionary of parameters to pass to the generate function
        kwargs: any additional arguments to pass to the generate function
        """
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

        self.model = None
        self.tokenizer = None

    def load(self, **kwargs):
        # Load the model and tokenizer
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)

        # cast model to fp16 and move to gpu if available
        if torch.cuda.is_available():
            self.model = self.model.half()
            self.model.cuda()
        
        # check if there is a padding token, if not add one
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def infer(self, input_texts : List[str], 
              generate_params : Dict[str, Any], 
              **kwargs: Any) -> Any:
        """
        input_texts: a list of strings, each string is a prompt
        generate_params: a dictionary of parameters to pass to the generate function
        kwargs: any additional arguments to pass to the generate function
        returns: a list of strings, each string is a generated output
        """

        # model.generate, use the tokenizer to convert the output to text and kwds for gen arguments
        # assume input and output are batched
        inp_text = input_texts

        inps = self.tokenizer(inp_text, return_tensors="pt", padding=True).to(self.model.device)
        output_txt = self.model.generate(**inps, **generate_params)

        # if we need to decode the text
        if not "no_decode" in generate_params:
            output_txt = self.tokenizer.batch_decode(output_txt, skip_special_tokens=True)

        return output_txt
    
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
        # check if there is a padding token, if not add one
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})


    def infer(self, input_texts : List[str], 
              generate_params : Dict[str, Any], 
              **kwargs: Any) -> Any:
        """
        input_texts: a list of strings, each string is a prompt
        generate_params: a dictionary of parameters to pass to the generate function
        kwargs: any additional arguments to pass to the generate function
        returns: a list of strings, each string is a generated output
        """

        # use self.client to call the model
        # assume input and output are batched
        inp_text = input_texts
        inps = self.tokenizer(inp_text, return_tensors="pt", padding=True)
        # call infer
        logits, output_txt = triton_call(self.client, self.model_name, inps.input_ids, **generate_params)

        if not "no_decode" in generate_params:
            output_txt = self.tokenizer.batch_decode(output_txt, skip_special_tokens=True)

        # check if logits are needed
        if generate_params["return_logits"]:
            return output_txt, logits
        else:
            return output_txt
    
