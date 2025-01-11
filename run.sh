#!/bin/bash
# TODO this will not be constant, need to check what the pod id using "get"
RUNPOD_POD_ID="igdzrkinh6bdzl"

source .venv/bin/activate
python rooporter.py

wall "shutdown"
api_key=$(<runpod_api_key)
runpodctl config --apiKey $api_key
runpodctl stop pod $RUNPOD_POD_ID
