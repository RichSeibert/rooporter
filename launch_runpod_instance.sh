# install runpod with: wget -qO- cli.runpod.net | sudo bash
# setup api key before running this with: runpodctl config --apiKey $RUNPOD_API_KEY
RUNPOD_POD_ID="igdzrkinh6bdzl"

runpodctl start pod $RUNPOD_POD_ID
POD_DETAILS=$(runpodctl get pod "$RUNPOD_POD_ID" --allfields)
ip_and_port=$(echo "$POD_DETAILS" | grep -oP '\d{1,3}(\.\d{1,3}){3}:\d+->22\xa0\(pub,tcp\)' | head -n 1)
ip=$(echo "$ip_and_port" | cut -d':' -f1)
port=$(echo "$ip_and_port" | cut -d':' -f2 | cut -d'-' -f1)
# TODO check if ntlk data needs to be downloaded every run (and if anything else needs to be setup which has been deleted after shutting down)
ssh -p $port -i ~/.ssh/runpod_key root@$ip "cd /workspace/rooporter; bash run.sh shutdown_after_run"
#runpodctl stop pod $RUNPOD_POD_ID
