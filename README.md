# rooporter

## setup
Run `bash setup.sh` and then activate venv with `source .venv/bin/activate`

## steps
0. Script is kicked off by cron job which is on a runpod instance. The runpod instance is started before the cronjob starts via a script which will run on a local machine, or something like a cheap AWS instance.
1. Fetch news articles
2. Generate one sentence summary and multi-sentence summary
3. Generate voice readover from multi-sentence summary
4. Generate videos based on one sentence summary. Keep generating until there is enough to conver length of readover
5. Combine video and readover
6. Upload to tiktok/youtube shorts

## example prompt for hunyuan. video-length must be a multiple of 4 + 1
`python sample_video.py --video-size 960 544 --video-length 129 --infer-steps 50 --prompt "President Joe Biden walks into a crowded auditorium wearing a traditional navy blue suit and a red tie, his suit jacket draped over his shoulder, a hint of a smile on his face. He steps up to the podium, adjusting the mic stand as he scans the audience with a hint of confidence." --flow-reverse --use-cpu-offload --save-path ./results`
