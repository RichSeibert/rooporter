PORT="22070"
IP="69.30.85.73"
rsync -vz -e "ssh -p $PORT -i ~/.ssh/runpod_key" /home/rich/Documents/rooporter/youtube_client_secret.json /home/rich/Documents/rooporter/credentials.pkl /home/rich/Documents/runpod_api_key root@$IP:/workspace/rooporter/
