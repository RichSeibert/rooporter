#!/bin/bash

# TODO figure out how to save this off into workspace dir so it's not deleted
if grep -q "Red Hat" /etc/redhat-release; then
    sudo dnf install vim tmux git cmake g++ htop -y
else
    apt update
    apt upgrade -y
    apt install vim tmux git cmake g++ htop -y
fi

if [ ! -d ".venv" ]; then
    python -m venv .venv
    pip install --upgrade pip wheel setuptools
    # TODO torch and flash-attn still need to be install manually
    pip install -r requirements.txt
fi
source .venv/bin/activate

if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    # Comment out cuda flag if not using GPU
    cmake -B build -DGGML_CUDA=ON
    # Change "-j" arg to number of cores
    cmake --build build --config Release -j 8
    cd ..
fi

if [ ! -d "models" ]; then
    mkdir models
    huggingface-cli download bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF --local-dir ./models
fi

if [ ! -d "MeloTTS" ]; then
    git clone https://github.com/myshell-ai/MeloTTS.git
    cd MeloTTS
    pip install -e .
    python -m unidic download
    cd ..
fi

if [ ! -d "HunyuanVideo" ]; then
    git clone https://github.com/Tencent/HunyuanVideo.git
    cd HunyuanVideo
    pip install requirements.txt
    huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
    cd ..
fi

