#!/bin/bash
#SBATCH -p g40
#SBATCH --account trlx
#SBATCH --gres=gpu:1
#SBATCH --output="%x.out"
#SBATCH --job-name=hf-infer
#SBATCH --exclusive

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# check if argument contains MODEL, NUMSHARD, PORT
if [ -z "$MODEL" ] || [ -z "$NUMSHARD" ] || [ -z "$PORT" ]; then
    echo "Please provide MODEL, NUM_SHARD, PORT"
    exit 1
fi

# check if CONDAENV is defined
if [ -z "$CONDAENV" ]; then
    echo "CONDAENV not found, defaulting to text-generation-inference."
    CONDA_ENV="text-generation-inference"
fi


# check if conda env text-generation-inference exists, if not run create_venv.sh
if ! conda env list | grep -q $CONDA_ENV; then
    ./create_venv.sh
fi

# activate bash rc
source ~/.bashrc
cd text-generation-inference

# activate conda env text-generation-inference
source ~/anaconda3/bin/activate $CONDA_ENV

# launch text-generation-inference server, using model_name as --model-id
text-generation-launcher --model-id $MODEL --num-shard $NUM_SHARD --port $PORT
