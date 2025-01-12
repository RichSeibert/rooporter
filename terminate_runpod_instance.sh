#!/bin/bash

echo "Deleting pod"
api_key=$(</home/rich/Documents/rooporter/runpod_api_key)
runpodctl config --apiKey $api_key
pod_id=$(runpodctl get pod | awk 'NR==2 {print $1}')
runpodctl remove pod $pod_id
