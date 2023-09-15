import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import torch
import transformers
import tritonclient.grpc.aio as grpcclient
from autocrit.inference.utils import best_of_n, triton_call
from text_generation import Client
from tqdm.asyncio import tqdm_asyncio


class InferenceHook(ABC):
    def __init__(self, **kwargs):
        """
        Args:
            kwargs (`Dict[str, Any]`): a dictionary of parameters to initilize the model with
        """
        pass

    @abstractmethod
    def generate(self, prompts: List[str], **kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Args:
            prompts (`List[str]`): inputs for generations
            kwargs (`Dict[str, Any]`): parameters to control generation

        Returns:
            outputs (`List[Dict[str, Any]]`): a list of dictionaries, each dictionary contains the following keys:
                id (`int`): the id of the prompt
                prompt (`str`): the prompt
                outputs (`List[str]`): a list of outputs per prompt
        """
        pass

    @abstractmethod
    def free(self):
        """
        Clean up resources after the inference
        """
        pass

class vLLMHook(InferenceHook):
    def __init__(self, model_path, tensor_parallel_size=1, num_external_nodes=0):
        """
        Starts data parallel vLLM servers either locally or on separate nodes by spawning slurm jobs

        Args:
            model_path (`str`): the path to the model
            tensor_parallel_size (`int`): the number of GPUs to use per one server
            num_external_nodes (`int`): spawn this many slurm jobs for the servers, if `0`, use only local resourses
        """
        self.init_time = time.time()
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.num_external_nodes = num_external_nodes
        self.nth_request = 0

        devices = list(map(str, range(torch.cuda.device_count())))
        devices = [",".join(devices[i*tensor_parallel_size:(i+1)*tensor_parallel_size]) for i in range(len(devices) // tensor_parallel_size)]

        if num_external_nodes:
            self.job_ids = []
            self.servers = []
            self.data_parallel_size = torch.cuda.device_count() * num_nodes // tensor_parallel_size

            sbatch_script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "vllm.sbatch")
            for _ in range(num_external_nodes):
                cmd = f"sbatch {sbatch_script_path} NUM_TP={tensor_parallel_size} MODEL_PATH={model_path} DEVICES={'|'.join(devices)}"
                process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, env={**os.environ, "TORCHELASTIC_USE_AGENT_STORE": ""})

                while True:
                    output = process.stdout.readline().decode("utf-8").strip()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output)
                        if output.startswith("Submitted batch job"):
                            self.job_ids.append(output.split()[-1].strip())

                while not os.path.exists(f"{self.job_ids[-1]}"):
                    time.sleep(1)

                with open(f"{self.job_ids[-1]}") as log:
                    while True:
                        output = log.readline().strip()
                        if output:
                            print(output)
                            if output.startswith("HOSTNAME="):
                                hostname = output.split("=")[-1].strip()
                                self.servers.extend([f"{hostname}:{8000+i}" for i in range(8 // tensor_parallel_size)])
                                break

        else:
            self.data_parallel_size = torch.cuda.device_count() // tensor_parallel_size
            self.servers = [f"localhost:{8000+i}" for i in range(self.data_parallel_size)]
            self.processes = []
            for i in range(self.data_parallel_size):
                cmd = f"python -m vllm.entrypoints.api_server -tp={tensor_parallel_size} --model={model_path} --port {8000+i}"
                kwargs = {"env": {**os.environ, "CUDA_VISIBLE_DEVICES": devices[i], "TORCHELASTIC_USE_AGENT_STORE": ""}}
                if not os.environ.get("DEBUG", False):
                    kwargs["stdout"] = subprocess.DEVNULL
                    kwargs["stderr"] = subprocess.DEVNULL

                process = subprocess.Popen(cmd.split(), **kwargs)
                self.processes.append(process)

            print(f"Loading {self.data_parallel_size} processes for {model_path}...")

        not_loaded = list(self.servers)
        while not_loaded:
            for server in not_loaded:
                try:
                    asyncio.run(self.request_vllm_api(server=server, prompt=".", max_new_tokens=1))
                    not_loaded.remove(server)
                except aiohttp.client_exceptions.ClientConnectorError:
                    break

            time.sleep(1)

        self.load_time = time.time() - self.init_time
        print(f"Loaded {model_path} in {self.load_time:.0f}s")

    async def request_vllm_api(self, prompt: str, i=0, num_return_sequences=1, temperature=0.0, max_new_tokens=512, stop=[], server=None):
        pload = {
            "prompt": prompt,
            "n": num_return_sequences,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "stop": stop,
            "stream": False,
        }

        if server is None:
            server = self.servers[self.nth_request % self.data_parallel_size]
            self.nth_request += 1

        connector = aiohttp.TCPConnector(limit_per_host=32768)
        timeout = aiohttp.ClientTimeout(total=3600)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.post(f"http://{server}/generate", json=pload) as response:
                try:
                    data = await response.json()
                    return {"id": i, "prompt": prompt, "outputs": [x[len(prompt):] for x in data["text"]]}
                except aiohttp.client.ContentTypeError:
                    return {"id": i, "prompt": prompt, "outputs": [None]}


    def generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        async def generate_vllm_api(prompts, **kwargs):
            outputs = [self.request_vllm_api(prompt=prompt, i=i, **kwargs) for i, prompt in enumerate(prompts)]
            return await tqdm_asyncio.gather(*outputs, desc=f"Inferencing {self.model_path}")

        batch_size = 32768
        outputs = []
        for i in range(0, len(prompts), batch_size):
            outputs += asyncio.run(generate_vllm_api(prompts[i:i+batch_size], **kwargs))

        return outputs

    def free(self):
        if self.num_external_nodes:
            if self.job_ids:
                subprocess.run(f"scancel {' '.join(self.job_ids)}".split())
                self.job_ids = []
        else:
            for p in self.processes:
                os.kill(p.pid, signal.SIGTERM)
                p.communicate()
            print(f"Unloaded all {self.model_path} processes")
            self.processes = []

    def __del__(self):
        self.free()

class HuggingFaceHook(InferenceHook):
    """
    Inference hook that uses plain HuggingFace transformers API
    """

    def __init__(self, model_path: str, tokenizer_path : Optional[str] = None):
        """
        Args:
            model_path (`str`): the directory of the model
            tokenizer_path (`str`): the directory of the tokenizer, if None, the `model_path` is implied
        """
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

        num_return_sequences = kwargs.get("num_return_sequences", 1)
        outputs = [{"id": i, "prompt": p, "outputs": outputs[i*num_return_sequences:(i+1)*num_return_sequences]} for i, p in enumerate(prompts)]

        return outputs

    def free(self):
        del self.model
        del self.tokenizer

class HuggingFaceHookBestOfN(HuggingFaceHook):
    """
    Inference hook that uses the HuggingFace API to call a model. Uses the best of N sampling method
    """

    def __init__(self, model_path: str, tokenizer_path : Optional[str] = None):
        """
        Args:
            model_path (`str`): the directory of the model
            tokenizer_path (`str`): the directory of the tokenizer, if None, the `model_path` is implied
        """
        super().__init__(model_path, tokenizer_path)

    def generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """
        Args:
            prompts: a list of prompts to generate
            kwargs: a dictionary of parameters to pass to the generate function
        """

        output_txt = best_of_n(self.model, self.tokenizer, prompts, gen_kwargs=kwargs)
        return output_txt


class TritonHook(InferenceHook):
    def __init__(self, url: str, model_path: str, tokenizer_path : Optional[str] = None):
        """
        Args:
            url (`str`): location of the triton server
            model_path (`str`): the name of the model to use
            tokenizer_path (`str`): the name of the tokenizer to use, if None, the `model_path` is implied
        """
        self.url = url # url contains host:port
        # TODO: if URL is a path to a triton model, we shold load the model and launch the server
        self.model_path = model_path
        if tokenizer_path is None:
            self.tokenizer_path = model_path
        else:
            self.tokenizer_path = tokenizer_path

        # create a client using url
        self.client = grpcclient.InferenceServerClient(url=self.url)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_path)
        # check if there is a padding token, if not add one
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})


    def generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """
        Args:
            prompts: a list of prompts to generate
            kwargs: a dictionary of parameters to pass to the generate function
        """

        # use self.client to call the model
        # assume input and output are batched
        inps = self.tokenizer(prompts, return_tensors="pt", padding=True)
        # call infer
        logits, output_txt = triton_call(self.client, self.model_path, inps.input_ids, **kwargs)

        if not "no_decode" in kwargs:
            output_txt = self.tokenizer.batch_decode(output_txt, skip_special_tokens=True)

        # check if logits are needed
        if kwargs["return_logits"]:
            return output_txt, logits
        else:
            return output_txt


class TextGenerationHook(InferenceHook):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # get num shards and port, used for the launcher script
        num_shards = kwargs.get("num_shards", 1)
        port = kwargs.get("port", 8080)

        # check if model name is a URL
        if not self.model_path.startswith("http"):
            # launch the model using the model name and text_generation_launcher.sh
            # The following line runs the launcher script
            output = subprocess.run(["sbatch./launch.sbatch MODEL_NAME="+str(self.model_path) + " NUM_SHARD="+str(num_shards) + " PORT="+str(port)], capture_output=True)
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
            self.client = Client(self.model_path)

    def generate(self, prompts: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        """
        Args:
            prompts (`List[str]`): a list of strings, each string is a prompt
            kwargs (`Dict[str, Any]`): a dictionary of parameters to pass to the generate function
        """
        # if input_texts is a list, convert it to a string
        if isinstance(input_texts, list):
            input_texts = input_texts[0]

        # use self.client to call the model
        output_txt = self.client.generate(input_texts, **kwargs).generated_text
        return output_txt
