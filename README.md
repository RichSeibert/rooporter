# rooporter

## TODO

## About
This project creates AI generated videos and uploads them to YouTube. There are two modes which are explained below.

This project can be ran standalone manually, but can also be setup with the `[bigkahuna](https://github.com/RichSeibert/bigkahuna)` project to run automatically on a schedule.

## Mode Descriptions
Move 0: Specific topic videos. Fill out the `mode_0_config.yaml` with prompts for video, audio, music, and YouTube title
Mode 1: Scrapes CNN news articles and creates narration videos. No manual configuration required like mode 0.

### Mode 0 Breakdown
1. Generate video and music prompts from an input prompt
2. Use hunyuan to generate videos
3. Use stable-audio to generate music
4. (Optional) generate voice over with Kokoro
4. Compile it all together

### Mode 1 Breakdown
1. Fetch news articles
2. Generate one sentence summary and multi-sentence summary
3. Generate voice readover from multi-sentence summary
4. Generate videos based on one sentence summary. Keep generating until there is enough to conver length of readover
5. Combine video and readover
6. Upload to tiktok/youtube shorts

## GPU machine setup
1. Transfer required files to instance with `transfer_files_to_runpod.sh <port> <ip>`
2. Login to huggingface-cli with "huggingface-cli login". Run `bash setup.sh`. For a fresh install, it will install all required packages/files
3. Update `config.ini` with settings that work with instance.

## Host machine setup
1. Setup aws instance and clone big_kahuna repo
2. Follow the README steps

## Example prompt for hunyuan. video-length must be a multiple of 4 + 1
`python sample_video.py --video-size 960 544 --video-length 129 --infer-steps 50 --prompt "President Joe Biden walks into a crowded auditorium wearing a traditional navy blue suit and a red tie, his suit jacket draped over his shoulder, a hint of a smile on his face. He steps up to the podium, adjusting the mic stand as he scans the audience with a hint of confidence." --flow-reverse --use-cpu-offload --save-path ./results`

### Using fp8 model instead
`python sample_video.py --video-size 960 544 --video-length 241 --infer-steps 35 --use-fp8 --dit-weight "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt" --prompt "President Joe Biden walks into a crowded auditorium wearing a traditional navy blue suit and a red tie, his suit jacket draped over his shoulder, a hint of a smile on his face. He steps up to the podium, adjusting the mic stand as he scans the audience with a hint of confidence." --flow-reverse
--use-cpu-offload --save-path ./results`
