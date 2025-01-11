#!/bin/bash

# TODO this will not be constant, need to check what the pod id using "get"
# TODO check if ntlk data needs to be downloaded every run (and if anything else needs to be setup as well after pod termination)

bash setup.sh

echo "Starting rooporter.py"
source .venv/bin/activate
python rooporter.py

# TODO get this to work. config fails while running inside of pod. Made terminate_runpod_instance.sh for now. That script can be deleted once this section is fixed
echo "Deleting pod"
api_key=$(<runpod_api_key)
runpodctl config --apiKey $api_key
pod_id=$(runpodctl get pod | awk 'NR==2 {print $1}')
runpodctl remove pod $pod_id
