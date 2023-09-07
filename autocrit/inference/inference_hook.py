import signal
import aiohttp
import asyncio
import json
import subprocess
import torch
import time
from tqdm.asyncio import tqdm_asyncio

from abc import ABC, abstractmethod
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


class InferenceHook(ABC):
    def __init__(self, **kwargs):
        """
        kwargs: a dictionary of parameters to pass to initilize the model
        """
        pass

    @abstractmethod
    def generate(self, prompts: List[str], **kwargs: Dict[str, Any]):
        """
        prompts: a list of strings, each string is a prompt
        kwargs: a dictionary of parameters to pass to the generate function
        """
        pass

    @abstractmethod
    def unload(self):
        pass

class vLLMHook(InferenceHook):
    def __init__(self, model_path, tensor_parallel_size=1):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = torch.cuda.device_count() // tensor_parallel_size
        self.nth_request = 0

        devices = list(map(str, range(torch.cuda.device_count())))
        devices = [",".join(devices[i*tensor_parallel_size:(i+1)*tensor_parallel_size]) for i in range(self.data_parallel_size)]

        self.processes = []
        for i in range(self.data_parallel_size):
            cmd = f"python -m vllm.entrypoints.api_server -tp={tensor_parallel_size} --model={model_path} --port {8000+i}"
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ, "CUDA_VISIBLE_DEVICES": devices[i]})
            self.processes.append(process)

        print(f"Loading {self.data_parallel_size} processes for {model_path}...")

        while True:
            all_loaded = True
            try:
                asyncio.run(self.request_vllm_api(prompt=""))
            except aiohttp.client_exceptions.ClientConnectorError:
                all_loaded = False

            if all_loaded:
                print(f"Loaded {model_path}")
                time.sleep(5)
                break

            time.sleep(1)

    async def request_vllm_api(self, prompt: str, i=0, n=1, temperature=0.0, max_new_tokens=512, stop=[]):
        pload = {
            "prompt": prompt,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "stop": stop,
            "stream": False,
        }

        port = 8000 + self.nth_request % self.data_parallel_size
        self.nth_request += 1
        connector = aiohttp.TCPConnector(limit_per_host=1024)
        timeout = aiohttp.ClientTimeout(total=9000)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.post(f"http://localhost:{port}/generate", json=pload) as response:
                try:
                    data = await response.json()
                    return {"id": i, "prompt": prompt, "output": [x[len(prompt):] for x in data["text"]]}
                except aiohttp.client.ContentTypeError:
                    return {"id": i, "prompt": prompt, "output": [None]}


    def generate(self, prompts, **kwargs):
        async def generate_vllm_api(prompts, **kwargs):
            outputs = [self.request_vllm_api(prompt, i=i, **kwargs) for i, prompt in enumerate(prompts)]
            return await tqdm_asyncio.gather(*outputs, desc=f"Inferencing {self.model_path}")

        return asyncio.run(generate_vllm_api(prompts, **kwargs))

    def unload(self):
        for p in self.processes:
            os.kill(p.pid, signal.SIGKILL)
            p.communicate()
        print(f"Offloaded all {self.model_path} processes")


# Inference hook that uses the HuggingFace API to call a model
class HuggingFaceHook(InferenceHook):
    def __init__(self, model_path: str, tokenizer_path : Optional[str] = None):
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    @torch.inference_mode()
    def generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        stop = kwargs.pop("stop", [])

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=kwargs.get("max_length", 2048)).to(self.model.device)

        all_ids = self.model.generate(**inputs, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, **kwargs)
        output_ids = all_ids[:, inputs.input_ids.shape[1]:]

        if "no_decode" in kwargs:
            return output

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for i in range(len(outputs)):
            for s in stop:
                if s in outputs[i]:
                    outputs[i] = outputs[i][:outputs[i].index(s)]

        return outputs

    def unload(self):
        del self.model
        del self.tokenizer

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
        # if input_texts is a list, convert it to a string
        if isinstance(input_texts, list):
            input_texts = input_texts[0]

        # use self.client to call the model
        output_txt = self.client.generate(input_texts, **generate_params).generated_text
        #print(output_txt)
        return output_txt
