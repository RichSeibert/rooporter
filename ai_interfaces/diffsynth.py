import torch
import torch.multiprocessing as mp
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData

def diffsynth_wan(prompt_with_id, num_frames, fps):
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            [
            "Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
            "Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
            "Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
            "Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
            "Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
            "Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
            ],
        "Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1-T2V-14B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.float8_e4m3fn, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

# Text-to-video
    video = pipe(
        prompt=prompt_with_id[1],
        negative_prompt="",
        num_inference_steps=50,
        seed=0,
        num_frames=num_frames,
        tiled=True
    )
    save_video(video, f"{0_prompt_with_id[0]}.mp4", fps=fps, quality=5)

def diffsynth_wan_multithread(prompts, num_frames, fps):
    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=1) as pool:
        prompt_with_ids = [[i, prompt] for i, prompt in enumerate(prompts)]
        results = pool.map(diffsynth_wan, prompt_with_ids, num_frames, fps)

