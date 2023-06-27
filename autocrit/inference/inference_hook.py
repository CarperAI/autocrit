import argparse
import os
import torch
import transformers
from typing import Tuple, Any, Optional, List, Dict
import tritonclient.grpc.aio as grpcclient
from autocrit.inference.utils import triton_call, best_of_n
from text_generation import Client
import logging  

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
        """
        dir: the directory of the model
        tokenizer_name: the name of the tokenizer to use, if None, use the model name
        """
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

        output_txt = self.model.generate(input_ids=inps.input_ids, attention_mask=inps.attention_mask, **generate_params)

        # if we need to decode the text
        if not "no_decode" in kwargs:
            output_txt = self.tokenizer.batch_decode(output_txt, skip_special_tokens=True)

        return output_txt

# Inference hook that uses the HuggingFace API to call a model. Uses the best of N sampling method
class HuggingFaceHookBestOfN(HuggingFaceHook):
    def __init__(self, dir : str, tokenizer_name : Optional[str] = None):
        """
        dir: the directory of the model
        tokenizer_name: the name of the tokenizer to use, if None, use the model name
        """
        super().__init__(dir, tokenizer_name)

    def infer(self, input_texts : List[str], 
              generate_params : Dict[str, Any], 
              **kwargs: Any) -> Any:
        """
        input_texts: a list of strings, each string is a prompt
        generate_params: a dictionary of parameters to pass to the generate function
        kwargs: any additional arguments to pass to the generate function
        returns: a list of strings, each string is a generated output
        """

        output_txt = best_of_n(self.model, self.tokenizer, input_texts, gen_kwargs=generate_params, **kwargs)
        return output_txt


class TritonHook(InferenceHook):
    def __init__(self, dir : str, model_name : str, tokenizer_name : Optional[str] = None):
        """
        dir: location of the triton server
        model_name: the name of the model to use
        tokenizer_name: the name of the tokenizer to use, if None, use the model name
        """
        super().__init__(dir)
        self.url = dir # url contains host:port
        # TODO: if URL is a path to a triton model, we shold load the model and launch the server
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

        if not "no_decode" in kwargs:
            output_txt = self.tokenizer.batch_decode(output_txt, skip_special_tokens=True)

        # check if logits are needed
        if generate_params["return_logits"]:
            return output_txt, logits
        else:
            return output_txt
    

# Inference hook that uses the HuggingFace API to call a model
class TextGenerationHook(InferenceHook):
    def __init__(self, dir : str):
        super().__init__(dir)
        self.model_name = dir
        if tokenizer_name is None:
            self.tokenizer_name = dir
        else:
            self.tokenizer_name = tokenizer_name

        self.client = None

    def load(self, **kwargs):
        # get num shards and port, used for the launcher script
        num_shards = kwargs.get("num_shards", 1)
        port = kwargs.get("port", 8080)

        # check if model name is a URL
            if not self.model_name.startswith("http"):
            # launch the model using the model name and text_generation_launcher.sh
            # The following line runs the launcher script
            output = subprocess.run(["sbatch./launch.sbatch MODEL_NAME="+str(self.model_name) + " NUM_SHARD="+str(num_shards) + " PORT="+str(port)], capture_output=True)
            logging.info(output.stdout.decode("utf-8"))
            # check return code
            if output.returncode != 0:
                logging.log(logging.ERROR, output.stderr.decode("utf-8"))
                raise RuntimeError("Failed to launch model")
            else: 
                logging.info("Model launched successfully.")

            # get the job id from the output
            job_id = output.stdout.decode("utf-8").split(" ")[-1].strip()
            logging.info("Job ID: " + str(job_id))

            # run scontrol to get the ip address
            output = subprocess.run(["scontrol show job " + str(job_id)], capture_output=True)

            # get the host ip address from the output. It looks like this: BatchHost=ip-xx-x-xxx-xxx
            ip = output.stdout.decode("utf-8").split("BatchHost=")[-1].strip()
            # convert it into a useable ip address
            ip = ip.replace("ip-", "").replace("-", ".")

            # Create the client
            self.client = Client(f"http://{ip}:{port}")
        else:
            self.client = Client(self.model_name)
                
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
        output_txt = self.client.generate(input_texts, **generate_params).tokens.text
        return output_txt
