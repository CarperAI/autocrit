#!/bin/bash
#SBATCH -p g40
#SBATCH --account trlx
#SBATCH --gres=gpu:1
#SBATCH --output="%x.out"
#SBATCH --job-name=hf-infer


# Check if the number of arguments is correct
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./text_generation_launcher.sh <model_name> <num_shards> <port>"
    exit 1
fi

# take the model name as a parameter
model_name=$1
# take number of shards as a parameter
num_shards=$2
# take port number as a parameter
port=$3



# check if text-generation-inference dir exists, if not clone the repo
if [ ! -d "text-generation-inference" ]; then
    git clone https://github.com/huggingface/text-generation-inference.git

conda activate text-generation-inference
cd text-generation-inference

# launch text-generation-inference server, using model_name as --model-id
text-generaiton-launcher --model-id $model_name --num-shard $num_shards --port $3
