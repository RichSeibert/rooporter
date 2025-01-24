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
import uuid
import re

import requests
from bs4 import BeautifulSoup

import pickle
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# TODO this is shit, shouldn't have to modify the path for melo and hunyuan, but setup.py doesn't work
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'MeloTTS')))
from melo.api import TTS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'HunyuanVideo')))
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

def generate_videos(audio_id_to_videos_generation_data):
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
    args.dit_weight = "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
    args.use_fp8 = True

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    # Get the updated args
    args = hunyuan_video_sampler.args

    audio_to_video_files = {}
    for audio_id, videos_to_generate_data in audio_id_to_videos_generation_data.items():
        audio_to_video_files[audio_id] = []
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
                audio_to_video_files[audio_id].append(audio_id + '_' + str(sub_id))
            except Exception as e:
                logging.error(f"Failed to generate video: {e}")

    os.chdir("..")
    return audio_to_video_files

def process_videos_and_audio(audio_video_mapping, output_file_name):
    logging.info("Processing videos and audio")
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
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(temp_list_file),
            "-c", "copy",
            str(combined_video)
        ]
        subprocess.run(combine_command, check=True)

        # Add the audio to the combined video
        audio_video_output = Path(f"output_{audio_file.stem}.mp4")
        add_audio = [
            "ffmpeg",
            "-y",
            "-i", str(combined_video),
            "-i", str(audio_file),
            "-c:v", "copy",
            "-c:a", "aac",
            "-strict", "experimental",
            str(audio_video_output)
        ]
        subprocess.run(add_audio, check=True)

        # Clean up intermediate video and add to the list for final combination
        combined_video.unlink()  # Remove intermediate combined video
        intermediate_videos.append(audio_video_output)

        # Clean up temporary list file
        temp_list_file.unlink()

    # Combine all processed videos into one final output
    final_list_file = Path("final_video_list.txt")
    with final_list_file.open("w") as f:
        f.write("file 'intro_video/intro_video_lower_volume.mp4'\n")
        for video in intermediate_videos:
            f.write(f"file '{video.resolve()}'\n")

    combine_all_command = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(final_list_file),
        "-c:v", "copy",
        "-crf", "18",
        "-y", f"tmp/{output_file_name}.mp4"
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
                 "-p", complete_prompt,
                 "-no-cnv"],
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
    """
    Scrape headlines and articles from CNN.
    
    Args:
        base_url (str): The base URL of CNN.
        limit (int): Number of articles to fetch.

    Returns:
        list of dict: List containing headline, article text, and URL.
    """
    logging.info(f"Scrape {base_url}")
    response = requests.get(base_url)
    
    if response.status_code != 200:
        logging.error(f"Failed to retrieve CNN homepage. Status code: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    logging.debug(f"Website html content - {soup}")
    articles = []
    
    # Find article links matching the pattern
    article_id = 0
    seen_hrefs = set()
    possible_article_data = soup.find_all('a', re.compile("container__link container__link--type-article*"), href=True)
    for link in possible_article_data:
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
                
    if len(articles) != limit:
        logging.warning(f"Didn't find enough articles to meet article limit - {limit}")
        logging.warning(f"All scraped data - {possible_article_data}")
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

def upload_to_youtube(video_path, title, description="", tags=None, category_id="25"):
    logging.info("Uploading to youtube")
    """
    Uploads an MP4 video to YouTube Shorts using the YouTube Data API v3.

    Args:
        video_path (str): Path to the MP4 video file.
        title (str): Title of the video.
        description (str): Description of the video.
        tags (list): List of tags for the video.
        category_id (str): Category ID of the video. 25 = news/politics

    Returns:
        dict: The YouTube API response.
    """
    # Authenticate using OAuth 2.0
    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    credentials = None
    try:
        with open("credentials.pkl", "rb") as token:
            credentials = pickle.load(token)
    except FileNotFoundError:
        logging.info("No saved credentials found, will generate them")

    if not credentials or not credentials.valid:
        logging.info("Credentials don't exist or are not valid, refreshing them")
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            logging.error("Need to authenticate through browser, exiting")
            return
            #flow = InstalledAppFlow.from_client_secrets_file("youtube_client_secret.json", SCOPES)
            #credentials = flow.run_local_server(port=0)

        with open("credentials.pkl", "wb") as token:
            pickle.dump(credentials, token)
            logging.info("Credentials saved to credentials.pkl")

    try:
        youtube = build("youtube", "v3", credentials=credentials)
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags if tags else [],
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": "public",
            },
        }
        media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    except Exception as e:
        logging.error(f"Failed to create youtube request: {e}")
        return

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            logging.info(f"Uploaded {int(status.progress() * 100)}%...")
    logging.info(f"Upload finished. Response: {response}")
    return response

def date_string_mdy():
    now = datetime.now()
    day_suffix = get_day_suffix(now.day)
    formatted_date = now.strftime(f"%b {now.day}{day_suffix}, %Y")
    return formatted_date

def get_day_suffix(day):
    if 11 <= day <= 13:  # Special case for 11th, 12th, 13th
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

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

# TODO move this into a different repo, so it can be used in multiple projects
# TODO this should not be called from within this script or another. It should
# should be in a standalone file so it can be ran from the command line start
# up commands. For example, "client.py register; roobot.py; rooporter.py; client.py terminate;"
class ManagerClient:
    def __init__(self):
        self.worker_id = str(uuid.uuid4())
        logging.info(f"Worker ID: {self.worker_id}")
        self.manager_url = "http://ec2-54-88-53-193.compute-1.amazonaws.com:8080"
        with open("token.txt") as file:
            token = file.read().split("\n")[0]
            self.header = {"Authorization": token}

    def register_with_manager(self):
        try:
            response = requests.post(f"{self.manager_url}/register-worker", json={"worker_id": self.worker_id}, headers=self.header)
            logging.info(f"Registration response: {response.json()}")
        except Exception as e:
            logging.error(f"Failed to register with manager: {e}")

    def notify_task_completed(self):
        try:
            response = requests.post(f"{self.manager_url}/task-completed", json={"worker_id": self.worker_id}, headers=self.header)
            logging.info(f"Task completion response: {response.json()}")
        except Exception as e:
            logging.error(f"Failed to notify manager: {e}")

def main():
    # TODO args cannot be used because they are picked up by the hunyuan video parser, and then an error will occur complaining about unrecognized args
    parser = argparse.ArgumentParser(description="Scrape news stories from CNN")
    parser.add_argument('--log', type=str, default='info', help="Log level (debug, info, warning, error, critical)")
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
		filename = 'logs/rooporter_' + year_month_date + '.log',
        format='%(asctime)s - %(levelname)s - %(message)s',
		datefmt='%H:%M:%S'
    )
    logging.info("\n-------------------------------------------------\n")


    # register with host server
    manager_client = ManagerClient()
    manager_client.register_with_manager()

    base_url = "https://www.cnn.com"
    try:
        articles = scrape_homepage(base_url, config_settings["number_of_articles"])
    except Exception as e:
        logging.critical(f"Failed to scrape website - {e}")
        manager_client.notify_task_completed()
        return
    if not articles:
        logging.critical("No articles scraped")
        manager_client.notify_task_completed()
        return

    # TODO this is shit
    # generate summarized version of each article
    article_texts = [a["text"] for a in articles]
    article_summary_prompt_info = PromptInfo(PromptInfo.article_summary,
                                             article_texts)
    for i, summary in enumerate(generate_text(article_summary_prompt_info,
                                              config_settings)):
        articles[i]["summary"] = summary

    # turn article summaries into audio
    try:
        articles_id_and_summary = [(article["id"], article["summary"]) for article in articles]
        generate_audio(articles_id_and_summary , config_settings["tts_worker_pool_size"])
    except Exception as e:
        logging.critical(f"Failed to generate audio - {e}")
        logging.info(f"Article data: {articles}")
        manager_client.notify_task_completed()
        return

    # generate video generation prompt for each article
    summarized_articles = [a["summary"] for a in articles] 
    video_prompt_prompt_info = PromptInfo(PromptInfo.video_prompt,
                                          summarized_articles)
    try:
        for i, video_prompt in enumerate(generate_text(video_prompt_prompt_info, config_settings)):
            articles[i]["video_prompt"] = video_prompt
    except Exception as e:
        logging.critical(f"Failed to generate video prompts - {e}")
        logging.info(f"Article data: {articles}")
        manager_client.notify_task_completed()
        return

    # Generate videos. Will generate multiple <video_duration> second videos 
    # for each audio clip to cover the entire clip and more. Output file format is <audio_file_id>_<video_file_id>.mp4
    audio_dir = "tmp/audio"
    audio_file_durations = get_wav_duration(audio_dir)
    video_generation_data = {}
    # TODO new model doesn't take as much ram. Test with hunyuan cli and see
    # if I can increase the duration. Only seems to hit 62% VRAM
    video_duration = 4
    for a in articles:
        id_s = str(a["id"])
        video_generation_data[id_s] = []
        num_videos = math.ceil(audio_file_durations[id_s]/video_duration)
        for sub_id in range(num_videos):
            # for the last video, make it so the combined durations of all 
            # videos is audio_duraiton + 1
            custom_duration = video_duration
            if sub_id == num_videos-1:
                if audio_file_durations[id_s] % video_duration == 0:
                    custom_duration += 1
                else:
                    custom_duration = (audio_file_durations[id_s] % video_duration) + 1
            ttv_output_file_name = id_s + "_" + str(sub_id)
            video_data = {"prompt": a["video_prompt"],
                          "duration": custom_duration,
                          "ttv_output_file_name": ttv_output_file_name}
            video_generation_data[id_s].append(video_data)

    try:
        audio_to_video_files = generate_videos(video_generation_data)
    except Exception as e:
        logging.critical(f"Failed to generate videos - {e}")
        logging.info(f"Video generation data: {video_generation_data}")
        manager_client.notify_task_completed()
        return

    # TODO add fade between each grouping of videos
    # TODO generate different prompt for each video instead of multiple videos from the same prompt
    # TODO add background music
    # TODO add subtitles for narration
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_file_name = f"finished_video_{time_stamp}"
    #audio_to_video_files = {"0": ["0_0", "0_1", "0_2", "0_3"], "1": ["1_0", "1_1", "1_2", "1_3"]}
    try:
        process_videos_and_audio(audio_to_video_files, output_file_name)
    except Exception as e:
        logging.critical(f"Failed to process videos and audio - {e}")
        logging.info(f"Audio and video files: {audio_to_video_files}")
        manager_client.notify_task_completed()
        return

	# TODO use https://github.com/makiisthenes/TiktokAutoUploader to copy youtube video to tiktok
    title = f"NEWS {date_string_mdy()}"
    try:
        upload_to_youtube("tmp/"+output_file_name+".mp4", title)
    except Exception as e:
        logging.critical(f"Failed to upload to youtube - {e}")
        manager_client.notify_task_completed()
        return

    # TODO add cleanup for logs and finished video files
    manager_client.notify_task_completed()

if __name__ == "__main__":
    main()
