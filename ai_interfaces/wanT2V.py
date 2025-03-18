import multiprocessing
import subprocess

def run_wan_script(prompt_with_id):
    command = [
        'python', 'generate.py',
        '--task', 't2v-14B',
        '--size', '1280*720',
        '--ckpt_dir', './Wan2.1-T2V-14B',
        '--save_file', f'{prompt_with_id[0]}.mp4',
        '--fps', '24',
        '--frame_num', '120',
        '--prompt', prompt_with_id[1]
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout, result.stderr

def wan_multithread(prompts):
    with multiprocessing.Pool(processes=2) as pool:
        results = pool.map(run_wan_script, [[i, prompt] for prompt in prompts])

    for stdout, stderr in results:
        if stdout:
            print("Output:", stdout)
        if stderr:
            print("Error:", stderr)
