import requests
from bs4 import BeautifulSoup
import logging
import argparse
import configparser
from datetime import datetime
from pathlib import Path
import subprocess
import os
import time
from multiprocessing import set_start_method, Pool
import wave
import math
import random

from melo.api import TTS

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args as hy_parse_args
from hyvideo.inference import HunyuanVideoSampler

class PromptInfo:
    article_summary = "article_summary"
    video_prompt = "video_prompt"
    def __init__(self, prompt_type, prompt_data):
        self.prompts = prompt_data
        if prompt_type == self.article_summary:
            self.system_prompt = "Summarize the input article into 1 sentence. You cannot write anything except for the article summary. Do not write something like 'Here is a summary of the article:', you can only write the summary."
            self.prompt_type = self.article_summary
        elif prompt_type == self.video_prompt:
            self.system_prompt = "Write a short, simple, descriptive, and funny 2 sentence scene of following article. Only describe the visuals of the scene. Do not write anything except for the prompt. Do not include the time duration of the video. Here is an example prompt: 'A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.'"
            self.prompt_type = self.video_prompt

def generate_videos(save_file_name, prompt_info):
    # TODO maybe just run the "sample_video.py" using subprocess. The repo is messed up and needs to have the whole setup.py thing, which is a pain
    logging.info("Generating videos")
    # have to change dir, some of the hunyuan scripts have hardcoded paths that expect the cwd to be HunyuanVideo
    os.chdir("HunyuanVideo")

    args = hy_parse_args() 
    args.prompt = prompt_info["prompt"]
    args.video_size = (960, 544)
    # determine video length from length of voiceover. Framerate is 24
    args.fps = 24
    args.video_length = (prompt_info["duration"] * args.fps) + 1
    args.seed = random.randint(1,999999)
    args.infer_steps = 30
    args.use_cpu_offload = True

    models_root_path = Path("./ckpts")
    if not models_root_path.exists():
        logging.error(f"Model directory `./ckpts` not found")
        return 1

    try:
        # Load models
        hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
        
        # Get the updated args
        args = hunyuan_video_sampler.args

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
                save_path = f"../tmp/video/{save_file_name}.mp4"
                save_videos_grid(sample, save_path, fps=args.fps)
                logging.info(f'Sample save to: {save_path}')
        os.chdir("..")
    except Exception as e:
        logging.error(f"Failed to generate video: {e}")
        os.chdir("..")
        return 1

def process_videos_and_audio(audio_video_mapping, output_file_name):
    logging.info("Processing videos and audio")
    """
    Combines video files associated with audio files, adds the audio, 
    and combines all the resulting videos into one.

    Args:
        audio_video_mapping (dict): Mapping where keys are audio file paths 
                                    and values are lists of associated video file paths.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    intermediate_videos = []

    # Process each audio and associated video files
    audio_path = "tmp/audio/"
    video_path = "tmp/video/"
    for audio_file, video_files in audio_video_mapping.items():
        audio_file = Path(audio_path+audio_file+".wav")
        video_files = [Path(video_path+video+".mp4") for video in video_files]
        
        # Check if all files exist
        for file in [audio_file] + video_files:
            if not file.exists():
                logging.error(f"Error: File not found: {file}")
                return False
        
        # Create a temporary file list for video concatenation
        temp_list_file = Path("video_list.txt")
        with temp_list_file.open("w") as f:
            for video in video_files:
                f.write(f"file '{video.resolve()}'\n")

        # Combine videos associated with the audio file
        combined_video = Path(f"combined_{audio_file.stem}.mp4")
        combine_command = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(temp_list_file),
            "-c", "copy",
            str(combined_video)
        ]
        subprocess.run(combine_command, check=True)

        # Add the audio to the combined video
        audio_video_output = Path(f"output_{audio_file.stem}.mp4")
        final_command = [
            "ffmpeg",
            "-i", str(combined_video),
            "-i", str(audio_file),
            "-c:v", "copy",
            "-c:a", "aac",
            "-strict", "experimental",
            str(audio_video_output)
        ]
        subprocess.run(final_command, check=True)

        # Clean up intermediate video and add to the list for final combination
        combined_video.unlink()  # Remove intermediate combined video
        intermediate_videos.append(audio_video_output)

        # Clean up temporary list file
        temp_list_file.unlink()

    # Combine all processed videos into one final output
    final_list_file = Path("final_video_list.txt")
    with final_list_file.open("w") as f:
        for video in intermediate_videos:
            f.write(f"file '{video.resolve()}'\n")

    combine_all_command = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", str(final_list_file),
        "-c", "copy",
        f"tmp/{output_file_name}.mp4"
    ]
    subprocess.run(combine_all_command, check=True)

    # Clean up intermediate videos and the final list file
    for video in intermediate_videos:
        video.unlink()
    final_list_file.unlink()

def generate_text(prompt_info, settings):
    logging.info(f"Generating text: {prompt_info.prompt_type}")
    llamaCpp_file_path = "llama.cpp/build/bin/llama-cli"
    model_file_path = "models/" + settings["model_file_name"]
    cpu_threads = str(settings["cpu_threads"])
    gpu_layers = str(settings["llama_cpp_gpu_layers"])
    context_len = "10000"

    outputs = []

    full_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Cutting Knowledge Date: December 2023
    Today Date: 26 Jul 2024

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    # TODO use 'popen' instead of 'run' for parallization
    for prompt in prompt_info.prompts:
        complete_prompt = full_prompt.replace("{system_prompt}", prompt_info.system_prompt)
        complete_prompt = complete_prompt.replace("{prompt}", prompt)
        try:
            result = subprocess.run(
                [llamaCpp_file_path,
                 "-m", model_file_path,
                 "-t", cpu_threads,
                 "-ngl", gpu_layers,
                 "--temp", "0.9",
                 "-c", context_len,
                 "-p", complete_prompt],
                capture_output=True,
                text=True
            )
            logging.debug(f"llama output: {result}")
            llm_output = result.stdout.strip()
            start_string = "assistant\n\n"
            start_idx = llm_output.find(start_string) + len(start_string)
            extra_ending = " [end of text]"
            llm_output_stripped = llm_output[start_idx:-len(extra_ending)]
            outputs.append(llm_output_stripped)
        except Exception as e:
            logging.error(f"Error running llama.cpp: {e}")
            outputs.append("")
            continue
    return outputs

def tts(data):
        file_name, text = data
        speed = 1.25
        # WARN - meloTTS doesn't clean up gpu memmory. Using multiprocess fixes
        # this and adds the benefit of parallization
        device = 'auto' # Will automatically use GPU if available
        model = TTS(language='EN', device=device)
        speaker_ids = model.hps.data.spk2id
        output_path = f"tmp/audio/{file_name}.wav"
        model.tts_to_file(text, speaker_ids['EN-US'], output_path, speed=speed)

def generate_audio(input_data, pool_size):
    logging.info("Generating audio")
    set_start_method("spawn", force=True)
    with Pool(processes=pool_size) as pool:
        pool.map(tts, input_data)

def parse_article(article_url):
    logging.info("Parsing article")
    # Fetch the article
    article_response = requests.get(article_url)
    if article_response.status_code != 200:
        logging.warning(f"Failed to fetch article at {article_url}")
        return

    article_soup = BeautifulSoup(article_response.content, 'html.parser')

    # Extract headline
    headline = article_soup.find('h1')
    if not headline:
        logging.warning(f"No headline found for {article_url}")
        return

    # Extract article text. Each paragraph is under a different class
    # , and the hyperlinks (hrefs) need to be removed
    paragraphs = article_soup.find_all('p', "paragraph inline-placeholder vossi-paragraph")
    article_text = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

    atricle = {}
    if article_text:
        article = {
            'headline': headline.get_text(strip=True),
            'text': article_text,
            'url': article_url
        }
        logging.info(f"Successfully scraped article: {article['url']}")

    return article

def scrape_homepage(base_url, limit=5):
    logging.info("Scrape website")
    """
    Scrape headlines and articles from CNN.
    
    Args:
        base_url (str): The base URL of CNN.
        limit (int): Number of articles to fetch.

    Returns:
        list of dict: List containing headline, article text, and URL.
    """
    logging.info("Starting scrape for CNN...")
    response = requests.get(base_url)
    
    if response.status_code != 200:
        logging.error(f"Failed to retrieve CNN homepage. Status code: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = []
    
    # Find article links matching the pattern
    article_id = 0
    seen_hrefs = set()
    for link in soup.find_all('a', "container__link container__link--type-article container_lead-package__link container_lead-package__left container_lead-package__light", href=True):
        href = link['href']
        if href in seen_hrefs:
            continue
        else:
            seen_hrefs.add(href)
            article_url = base_url + href

        try:
            article_data = parse_article(article_url)
            if article_data:
                articles.append(article_data)
                articles[-1]["id"] = article_id 
                article_id += 1
            if len(articles) >= limit:
                break
        except Exception as e:
            logging.error(f"Error while processing article at {article_url}: {e}")
                
    return articles

def get_wav_duration(directory):
    files = os.listdir(directory)
    audio_file_durations = {}
    for file in files:
        with wave.open(directory+'/'+file, 'r') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            id_ = file.split('.', 1)[0]
            audio_file_durations[id_] = math.ceil(frames / float(rate))
    return audio_file_durations

def parse_config(config):
    config_settings = {}
    try:
        config.read('config.ini')
        config_settings['tts_worker_pool_size'] = config.getint('DEFAULT', 'tts_worker_pool_size')
        config_settings['number_of_articles'] = config.getint('DEFAULT', 'number_of_articles')
        config_settings['cpu_threads'] = config.getint('LLAMACPP', 'cpu_threads')
        config_settings['llama_cpp_gpu_layers'] = config.getint('LLAMACPP', 'llama_cpp_gpu_layers')
        config_settings['model_file_name'] = config.get('LLAMACPP', 'model_file_name')
        return config_settings
    except Exception as e:
        logging.error(f"Config file issue: {e}")

def main():
    # TODO args cannot be used because they are picked up by the hunyuan video parser, and then an error will occur complaining about unrecognized args
    parser = argparse.ArgumentParser(description="Scrape news stories from CNN")
    parser.add_argument('--limit', type=int, default=5, help="Number of articles to scrape")
    parser.add_argument('--log', type=str, default='debug', help="Log level (debug, info, warning, error, critical)")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config_settings = parse_config(config)
    if not config_settings or not args:
        return

    # Set up logging
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    year_month_date = datetime.now().strftime("%Y_%m_%d")
    logging.basicConfig(
        level=log_level,
		filename = 'rooporter_' + year_month_date + '.log',
        format='%(asctime)s - %(levelname)s - %(message)s',
		datefmt='%H:%M:%S'
    )

    base_url = "https://www.cnn.com"
    articles = scrape_homepage(base_url, 
                               config_settings["number_of_articles"])

    # TODO this is shit
    # generate summarized version of each article
    article_texts = [a["text"] for a in articles]
    prompt_info = PromptInfo(PromptInfo.article_summary, 
                             article_texts)
    article_summary_prompt_info = prompt_info
    for i, summary in enumerate(generate_text(article_summary_prompt_info,
                                              config_settings)):
        articles[i]["summary"] = summary

    # turn article summaries into audio
    generate_audio([(article["id"], article["summary"]) for article in articles],
                   config_settings["tts_worker_pool_size"])

    # generate video generation prompt for each article
    summarized_articles = [a["summary"] for a in articles] 
    prompt_info = PromptInfo(PromptInfo.video_prompt, 
                             summarized_articles)
    video_prompt_prompt_info = prompt_info
    for i, video_prompt in enumerate(generate_text(video_prompt_prompt_info,
                                                   config_settings)):
        articles[i]["video_prompt"] = video_prompt
    logging.debug(f"Article data: {articles}")

    # Generate videos. Will generate multiple <video_duration> second videos for each audio clip to cover the entire clip and more. Output file format is <audio_file_id>_<video_file_id>.mp4
    audio_dir = "tmp/audio"
    audio_file_durations = get_wav_duration(audio_dir)
    video_prompts_info = {}
    video_duration = 4
    for a in articles:
        id_ = a["id"]
        video_prompts_info[id_] = {"prompt": a["video_prompt"],
                                   "duration": video_duration}
    audio_to_video_files = {}
    for id_, prompt_info in video_prompts_info.items():
        id_s = str(id_)
        audio_to_video_files[id_s] = []
        num_videos = math.ceil(audio_file_durations[id_s]/video_duration)
        for sub_id in range(num_videos):
            # for the last video, make it so the combined durations of all 
            # videos is audio_duraiton + 1
            if sub_id == num_videos-1:
                if audio_file_durations[id_s] % video_duration == 0:
                    prompt_info["duration"] += 1
                else:
                    prompt_info["duration"] = (audio_file_durations[id_s] % video_duration) + 1
            video_file_name = id_s + "_" + str(sub_id) 
            rc = generate_videos(video_file_name, prompt_info)
            if not rc:
                audio_to_video_files[id_s].append(video_file_name)

    # TODO add fade between each grouping of videos, and maybe include an intro video
    # TODO add subtitles for narration
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_file_name = f"finished_video_{time_stamp}"
    process_videos_and_audio(audio_to_video_files, output_file_name)

    # TODO upload

    # TODO delete everything in tmp/audio and tmp/video directories

if __name__ == "__main__":
    main()
