#!/bin/bash

# TODO figure out how to save off installed packages from apt/dnf into workspace dir so they are not deleted when the instance is terminated

MODE=-1
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE=$2
            shift 2
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done
if [ "$MODE" == "-1" ]; then
    echo "You must specify mode (-m, --mode)"
    exit 1
fi

source /etc/os-release
if [ "$DEV_MODE" == true  ]; then
    echo "Dev mode, installing extra packages for debug"
    # TODO this check is broken, just check if this dir exists
    if [ $ID_LIKE == "fedora" ]; then
        sudo dnf install ffmpeg vim tmux git cmake g++ htop rsync -y
    else
        apt update
        apt upgrade -y
        apt install ffmpeg vim tmux git cmake g++ htop rsync -y
        apt install --upgrade python3.10-venv -y
    fi
else
    echo "Installing packages"
    if [ $ID_LIKE == "fedora" ]; then
        sudo dnf install ffmpeg git -y
    else
        apt update
        apt upgrade -y
        apt install ffmpeg git cmake g++ -y
        apt install --upgrade python3.10-venv -y
    fi
fi

# TODO add git commands to pull newest commits. Will need to configure this so config is setup with git key, or other method
# if [ ! -d "/workspace/.my-credentials" ]; then
#   git config --global credential.helper 'store --file /workspace/.my-credentials'
# fi
# <command to load credentials using .my-credentials file>
# git pull

mkdir -p "tmp/audio" "tmp/video" "logs"

if [ ! -d ".venv" ]; then
    echo "Creating python virtual env"
    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip wheel setuptools
    # TODO torch and flash-attn don't install right when using requirements.txt
    pip install numpy
    pip install -r requirements.txt
    pip install torch
    pip install flash-attn
    pip install "huggingface_hub[cli]"
    pip install -q kokoro>=0.8.2 soundfile
    pip install stable-audio-tools
    apt-get -qq -y install espeak-ng > /dev/null 2>&1
else
    echo ".venv exists, just activate it"
    source .venv/bin/activate
fi

if [ ! -L "llama.cpp" ]; then
    echo "Creating llama.cpp"
    if [ ! -d "../llama.cpp" ]; then
        cd ..
        git clone https://github.com/ggerganov/llama.cpp.git
        cd llama.cpp
        # Comment out cuda flag if not using GPU
        cmake -B build -DGGML_CUDA=ON
        # Change "-j" arg to number of cores
        cmake --build build --config Release -j 8
        cd ..
        cd rooporter
    fi
    ln -s ../llama.cpp llama.cpp
fi

if [ ! -L "models" ]; then
    echo "Downloading models"
    if [ ! -d "../models" ]; then
        mkdir ../models
    fi
    huggingface-cli download bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF Llama-3.1-8B-Lexi-Uncensored-V2-Q8_0.gguf --local-dir ../models
    ln -s ../models models
fi

if [ "$MODE" == "0" ] || [ "$MODE" == "1" ]; then
    echo "Setting up Wan2.1, Stable-Audio, and Kokoro-82M for mode $MODE"
    if [ ! -L "Wan2.1" ]; then
        git clone https://github.com/Wan-Video/Wan2.1.git
        cd Wan2.1
        pip install -r requirements.txt
        cd ..
    fi
    if [ ! -L "Wan2.1-T2V-14B" ]; then
        huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ../models/Wan2.1-T2V-14B
    fi
    #huggingface-cli download stabilityai/stable-audio-open-1.0 --local-dir ../models/stable-audio-open1.0
    #huggingface-cli download hexgrad/Kokoro-82M --local-dir ../models/Kokoro-82M
elif [ "$MODE" == "2" ]; then
    echo "Setting up Hunyuan and MeloTTS for mode 2"
    if [ ! -d "HunyuanVideo" ]; then
        echo "Creating HunyuanVideo"
        git clone https://github.com/Tencent/HunyuanVideo.git
        cd HunyuanVideo
        pip install -r requirements.txt
        huggingface-cli download tencent/HunyuanVideo hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8_map.pt --local-dir ckpts
        cd ckpts
        huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./llava-llama-3-8b-v1_1-transformers
        cd ..
        python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir ckpts/llava-llama-3-8b-v1_1-transformers --output_dir ckpts/text_encoder
        cd ckpts
        huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2
        cd ..
    fi
    if [ ! -d "MeloTTS" ]; then
        echo "Creating MeloTTS"
        git clone https://github.com/myshell-ai/MeloTTS.git
        cd MeloTTS
        pip install -e .
        python -m unidic download
        cd ..
    fi
fi

