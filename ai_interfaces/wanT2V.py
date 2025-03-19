import multiprocessing
import subprocess

def run_wan_script(prompt_with_id):
    command = [
        'python', 'Wan2.1/generate.py',
        '--task', 't2v-14B',
        '--size', '1280*720',
        '--ckpt_dir', './Wan2.1-T2V-14B',
        '--save_file', f'/workspace/rooporter/tmp/videos/{prompt_with_id[0]}.mp4',
        #'--fps', '24', # TODO figure out how to set fps
        '--frame_num', '80',
        '--prompt', prompt_with_id[1]
    ]
    result = subprocess.run(command, text=True)

def wan_multithread(prompts):
    with multiprocessing.Pool(processes=1) as pool:
        results = pool.map(run_wan_script, [[i, prompt] for i, prompt in enumerate(prompts)])
