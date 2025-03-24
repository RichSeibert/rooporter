# rooporter

## TODO
1. Load audio model from local file (fix where hf saves it's cache, env variable not working)
3. Uncomment manager_client stuff once I get host back up and running
4. Redo the entire gpu instance, fucked up after wan shit

## Modes
0 = Specific topic videos 
1 = Quote videos
2 = CNN news and narration videos (Uses hunyuan for T2V and MeloTTS for T2S)

## Mode 0
1. Generate video and music prompts from an input prompt
2. Use Wan2.1 to generate videos
3. Use stable-audio to generate music
4. Compile it all together

## Mode 1
1. Generate video and music prompts from a quote
2. Generate voice over using kokoro
3. Use Wan2.1 to generate videos
4. Use stable-audio to generate music
5. Compile it all together

## Mode 2
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
1. Install runpodctl and setup config (instructions in `launch_runpod_instance.sh`)
2. Setup a cronjob to run `launch_runpod_instance.sh` and `terminate_runpod_instance.sh` a few hours later (at least until I can figure out how to terminate an instance from within itself)

## Example prompt for hunyuan. video-length must be a multiple of 4 + 1
`python sample_video.py --video-size 960 544 --video-length 129 --infer-steps 50 --prompt "President Joe Biden walks into a crowded auditorium wearing a traditional navy blue suit and a red tie, his suit jacket draped over his shoulder, a hint of a smile on his face. He steps up to the podium, adjusting the mic stand as he scans the audience with a hint of confidence." --flow-reverse --use-cpu-offload --save-path ./results`

### Using fp8 model instead
`python sample_video.py --video-size 960 544 --video-length 241 --infer-steps 35 --use-fp8 --dit-weight "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt" --prompt "President Joe Biden walks into a crowded auditorium wearing a traditional navy blue suit and a red tie, his suit jacket draped over his shoulder, a hint of a smile on his face. He steps up to the podium, adjusting the mic stand as he scans the audience with a hint of confidence." --flow-reverse
--use-cpu-offload --save-path ./results`
