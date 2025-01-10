# setup api key before running this with runpodctl config --apiKey $RUNPOD_API_KEY
RUNPOD_POD_ID = 123
PORT = "69.30.85.9"
IP = 22201
runpodctl start pod $RUNPOD_POD_ID
POD_DETAILS=$(runpodctl get pod "$RUNPOD_POD_ID" --output=json)
ssh -p $PORT -i ~/.ssh/runpod_key root@$IP "ls; pwd"
runpodctl stop pod $RUNPOD_POD_ID
