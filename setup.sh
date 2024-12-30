#!/bin/bash

if grep -q "Red Hat" /etc/redhat-release; then
    sudo dnf install vim tmux git cmake g++ htop -y
else
    sudo apt update
    sudo apt upgrade
    sudo apt install vim tmux git cmake g++ htop -y
fi

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
# Comment out cuda flag if not using GPU
cmake -B build -DGGML_CUDA=ON
# Change "-j" arg to number of cores
cmake --build build --config Release -j 8
cd ..

git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
python -m unidic download
cd ..

git clone https://github.com/Tencent/HunyuanVideo.git
cd HunyuanVideo
pip install requirements.txt
huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
cd ..

