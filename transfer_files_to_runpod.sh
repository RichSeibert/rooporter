#!/bin/bash

if [ $# -le 1 ]; then
    echo "Usage: transfer_files_to_runpod.sh [port] [ip]"
    exit 1
fi

rsync -vz -e "ssh -p $1 -i ~/.ssh/runpod_key" /home/rich/Documents/rooporter/credentials.pkl /home/rich/Documents/rooporter/tokens.yaml root@$2:/workspace/rooporter/
