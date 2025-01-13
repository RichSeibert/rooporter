#!/bin/bash

# install runpodctl with: wget -qO- cli.runpod.net | sudo bash
# setup api key before running this with: runpodctl config --apiKey $RUNPOD_API_KEY

runpodctl create pod --args "bash -c 'cd /workspace/rooporter; bash run.sh &> last_run_output.txt &; /start.sh'" \
                     --secureCloud \
                     --containerDiskSize 10 \
                     --gpuCount 1 \
                     --gpuType "NVIDIA A40" \
                     --templateId "runpod-torch-v240" \
                     --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
                     --name "rooporter_pod" \
                     --volumePath "/workspace" \
                     --ports "22/tcp" \
                     --networkVolumeId "ys3y7qzc5y"
