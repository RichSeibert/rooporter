#!/bin/bash

if [ $# -le 1 ]; then
    echo "Usage: transfer_files_to_runpod.sh [port] [ip]"
    exit 1
fi

# Remember to run "git update-index --assume-unchanged tokens.yaml" on gpu
# machine after the transfer
rsync -vz -e "ssh -p $1 -i ~/.ssh/runpod_key" /home/rich/Documents/rooporter/credentials.pkl /home/rich/Documents/rooporter/tokens.yaml root@$2:/workspace/rooporter/
