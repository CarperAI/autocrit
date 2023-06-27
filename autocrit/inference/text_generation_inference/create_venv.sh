#!/bin/bash
source ~/.bashrc

# check if conda env text-generation-inference exists, if not create it
if ! conda env list | grep -q "text-generation-inference"; then
    conda create -n text-generation-inference python=3.9
fi

# activate conda env text-generation-inference
conda activate text-generation-inference

# retrieve the path of the current conda env
CONDA_ENV_PATH=$(conda info --base)/envs/$(conda info --envs | grep "*" | awk '{print $1}')/

# update pip
python -m pip install --upgrade pip

# check if text-generation-inference dir exists, if not clone the repo
if [ ! -d "text-generation-inference" ]; then
    git clone https://github.com/huggingface/text-generation-inference.git
fi

# install text-generation-inference requirements
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
unzip -o $PROTOC_ZIP -d $CONDA_ENV_PATH bin/protoc
unzip -o $PROTOC_ZIP -d $CONDA_ENV_PATH 'include/*'
rm -f $PROTOC_ZIP

# install text-generation-inference
cd text-generation-inference
module load cuda/11.8
BUILD_EXTENSIONS=True make install # Install repos
