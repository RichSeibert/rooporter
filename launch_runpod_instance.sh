# install runpod with: wget -qO- cli.runpod.net | sudo bash
# setup api key before running this with: runpodctl config --apiKey $RUNPOD_API_KEY
RUNPOD_POD_ID="igdzrkinh6bdzl"

# TODO this will have to change to create pod and remove pod. Stopping a pod means that later on it might be taken by someone else
runpodctl start pod $RUNPOD_POD_ID
POD_DETAILS=$(runpodctl get pod "$RUNPOD_POD_ID" --allfields)
ip_and_port=$(echo "$POD_DETAILS" | grep -oP '\d{1,3}(\.\d{1,3}){3}:\d+->22\xa0\(pub,tcp\)' | head -n 1)
ip=$(echo "$ip_and_port" | cut -d':' -f1)
port=$(echo "$ip_and_port" | cut -d':' -f2 | cut -d'-' -f1)
echo $ip, $port
# TODO check if ntlk data needs to be downloaded every run (and if anything else needs to be setup which has been deleted after shutting down)
api_key=$(<runpod_api_key)
ssh -o "StrictHostKeyChecking no" -p $port -i ~/.ssh/runpod_key root@$ip "cd /workspace/rooporter; bash run.sh shutdown_after_run;"
