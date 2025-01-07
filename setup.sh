#!/bin/bash

# TODO figure out how to save this off into workspace dir so it's not deleted
# TODO this check is broken, just check if this dir exists
if grep -q "Red Hat" /etc/redhat-release; then
    sudo dnf install ffmpeg vim tmux git cmake g++ htop rsync -y
else
    apt update
    apt upgrade -y
    apt install ffmpeg vim tmux git cmake g++ htop rsync -y
fi

if [ ! -d ".venv" ]; then
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip wheel setuptools
    # TODO torch and flash-attn don't install right when using requirements.txt
    pip install -r requirements.txt
    pip install bs4
    pip install torch
    pip install torchvideo
    pip install flash-attn
    # this runs the setup.py script
    pip install -e .
else
    source .venv/bin/activate
    pip install -e .
fi

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
    huggingface-cli download bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF Llama-3.1-8B-Lexi-Uncensored-V2-Q8_0.gguf --local-dir ./models
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
    pip install -r requirements.txt
    huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
    cd ckpts
    huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./llava-llama-3-8b-v1_1-transformers
    cd ..
    python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir ckpts/llava-llama-3-8b-v1_1-transformers --output_dir ckpts/text_encoder
    cd ckpts
    huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2
    cd ..
fi

