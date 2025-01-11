#!/bin/bash

echo "Run setup.sh"
bash setup.sh

echo "Run rooporter.py"
source .venv/bin/activate
python rooporter.py

# TODO get this to work. config fails while running inside of pod. Made terminate_runpod_instance.sh for now. That script can be deleted once this section is fixed
echo "Deleting pod"
api_key=$(<runpod_api_key)
runpodctl config --apiKey $api_key
pod_id=$(runpodctl get pod | awk 'NR==2 {print $1}')
runpodctl remove pod $pod_id
