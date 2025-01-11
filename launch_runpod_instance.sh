#!/bin/bash

# install runpodctl with: wget -qO- cli.runpod.net | sudo bash
# setup api key before running this with: runpodctl config --apiKey $RUNPOD_API_KEY

runpodctl create pod --args "bash /workspace/rooporter/run.sh" --containerDiskSize 20 --gpuCount 1 --gpuType "A40" --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" --name "rooporter_pod" --ports "22/tcp" --volumePath "/workspace" --networkVolumeId 1
