#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --partition=g40
#SBATCH --account=stability
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --cpus-per-task=64
#SBATCH --output=vllm_logs/%j
#SBATCH --exclusive

ray stop

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# check if argument contains MODEL, NUMTP
if [ -z "$MODEL_PATH" ] || [ -z "$NUM_TP" ]; then
    echo "Please provide MODEL, NUM_TP"
    exit 1
fi

# replace '|' with ' ' for cuda devices separator to iterate over
export DEVICES=${DEVICES//|/ }
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

echo MODEL_PATH=$MODEL_PATH
echo NUM_TP=$NUM_TP
echo DEVICES=$DEVICES
echo HOSTNAME=$HOSTNAMES

echo $VIRTUAL_ENV
module load cuda/11.8

ix=0
for devices in $DEVICES; do
    CUDA_VISIBLE_DEVICES=$devices python -m vllm.entrypoints.api_server -tp=$NUM_TP --model=$MODEL_PATH --host=0.0.0.0 --port $((8000+ix)) &
    ix=$((ix+1))
done

wait
