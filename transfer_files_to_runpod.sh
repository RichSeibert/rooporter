PORT = 
IP = 
rsync -vz -e "ssh -p $PORT -i ~/.ssh/runpod_key" Documents/rooporter/youtube_client_secret.json Documents/rooporter/credentials.pkl root@$IP:/workspace/rooporter/
