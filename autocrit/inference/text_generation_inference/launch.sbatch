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


# check if conda env text-generation-inference exists, if not run create_venv.sh
if ! conda env list | grep -q "text-generation-inference"; then
    ./create_venv.sh
fi

# activate bash rc
source ~/.bashrc
cd text-generation-inference

# launch text-generation-inference server, using model_name as --model-id
text-generation-launcher --model-id $model_name --num-shard $num_shards --port $3
