import sys
import os
import logging
from pathlib import Path
# TODO this is shit, shouldn't have to modify the path for melo and hunyuan, but setup.py doesn't work
base_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(os.path.join(base_dir, 'HunyuanVideo')))
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args as hy_parse_args
from hyvideo.inference import HunyuanVideoSampler

def generate_videos_hunyuan(audio_id_to_videos_generation_data):
    logging.info("Generating videos")
    # have to change dir, some of the hunyuan scripts have hardcoded paths that expect the cwd to be HunyuanVideo
    os.chdir("HunyuanVideo")
    models_root_path = Path("./ckpts")
    if not models_root_path.exists():
        logging.critical(f"Model directory `./ckpts` not found")
        raise Exception("./ckpts dir not found in HunyuanVideo directory")

    args = hy_parse_args() 
    args.video_size = (960, 544)
    # determine video length from length of voiceover. Framerate is 24
    args.fps = 24
    args.infer_steps = 30
    args.use_cpu_offload = True
    args.embedded_cfg_scale = 6.0
    args.flow_shift = 7.0
    args.flow_reverse = True
    args.dit_weight = "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
    args.use_fp8 = True

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    # Get the updated args
    args = hunyuan_video_sampler.args

    audio_to_video_files = {}
    for audio_id, videos_to_generate_data in audio_id_to_videos_generation_data.items():
        audio_to_video_files[audio_id] = {"video_files": []}
        for sub_id, video_data in enumerate(videos_to_generate_data):
            args.prompt = video_data["prompt"]
            args.video_length = (video_data["duration"] * args.fps) + 1
            args.seed = random.randint(1,999999)
            logging.info(f"Generate video with prompt - {args.prompt}, and duration - {video_data['duration']} ")
            try:
                # Start sampling
                outputs = hunyuan_video_sampler.predict(
                    prompt=args.prompt,
                    height=args.video_size[0],
                    width=args.video_size[1],
                    video_length=args.video_length,
                    seed=args.seed,
                    negative_prompt=args.neg_prompt,
                    infer_steps=args.infer_steps,
                    guidance_scale=args.cfg_scale,
                    num_videos_per_prompt=args.num_videos,
                    flow_shift=args.flow_shift,
                    batch_size=args.batch_size,
                    embedded_guidance_scale=args.embedded_cfg_scale
                )
                samples = outputs['samples']
                # Save samples
                if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
                    for i, sample in enumerate(samples):
                        sample = samples[i].unsqueeze(0)
                        save_file_name = video_data["ttv_output_file_name"]
                        save_path = f"../tmp/video/{save_file_name}.mp4"
                        save_videos_grid(sample, save_path, fps=args.fps)
                        logging.info(f'Sample save to: {save_path}')
                audio_to_video_files[audio_id]["video_files"].append(audio_id + '_' + str(sub_id))
            except Exception as e:
                logging.error(f"Failed to generate video: {e}")

    os.chdir("..")
    return audio_to_video_files

