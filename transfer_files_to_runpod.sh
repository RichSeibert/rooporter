#!/bin/bash

if [ $# -le 1 ]; then
    echo "Usage: transfer_files_to_runpod.sh [port] [ip]"
    exit 1
fi

rsync -vz -e "ssh -p $1 -i ~/.ssh/runpod_key" /home/rich/Documents/rooporter/youtube_client_secret.json /home/rich/Documents/rooporter/credentials.pkl /home/rich/Documents/runpod_api_key /home/rich/Documents/rooporter/host_token.txt root@$2:/workspace/rooporter/
