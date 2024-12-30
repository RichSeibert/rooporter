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
pip install -r requirements.txt

git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
# Change number to core cound, and add cuda flag if using GPU
cmake --build build --config Release -j 2
cd ..

git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
python -m unidic download
cd ..
