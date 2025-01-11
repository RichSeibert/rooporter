#!/bin/bash

# TODO this will not be constant, need to check what the pod id using "get"
# TODO check if ntlk data needs to be downloaded every run (and if anything else needs to be setup as well after pod termination)

echo "running setup.sh"
bash setup.sh

echo "Starting rooporter.py"
source .venv/bin/activate
python rooporter.py

echo "Deleting pod"
api_key=$(<runpod_api_key)
runpodctl config --apiKey $api_key
pod_id=$(runpodctl get pod | awk 'NR==2 {print $1}')
runpodctl remove pod $pod_id
