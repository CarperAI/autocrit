To convert a casual language model to triton inference server format:
```bash
python convert_to_triton.py --model gpt2
```

To use an API client, see [client.py](./client.py).

Running `python client.py --url ip-26-0-130-12:8001` will run a inference REPL

To start a server using Docker:
`./start_server.sh`

To start a server on SLURM using enroot:
`srun --container-image nvcr.io#nvidia/tritonserver:23.01-py3 --container-mounts=/path/to/model_store:/models tritonserver --model-repository=/models`

or run the [example sbatch](./triton.sbatch)
```
sbatch triton.sbatch
```

Use `ip-hostname-of-node:8001` as the url 
