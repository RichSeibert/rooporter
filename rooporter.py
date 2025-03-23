#!/usr/bin/env python3.10

import logging
import argparse
import configparser
from datetime import datetime
from pathlib import Path
import subprocess
import os
import time
import wave
import math
import random
import uuid
import re
import yaml

import requests
from bs4 import BeautifulSoup

import pickle
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from ai_interfaces.llama_cpp import PromptInfo, generate_text

def process_videos_and_audio(audio_video_mapping, output_file_name):
    logging.info("Processing videos and audio")
    intermediate_videos = []

    # Process each audio and associated video files
    audio_path = "tmp/audio/"
    video_path = "tmp/video/"
    for audio_file, audio_file_data in audio_video_mapping.items():
        audio_file = Path(audio_path + audio_file + ".wav")
        video_files = [Path(video_path + video + ".mp4") for video in audio_file_data["video_files"]]

        # Check if all files exist
        for file in [audio_file] + video_files:
            if not file.exists():
                logging.error(f"Error: File not found: {file}")
                return False

        if "audio_duration" in audio_file_data:
            processed_audio = Path(f"trimmed_{audio_file.stem}.wav")
            trim_command = [
                "ffmpeg",
                "-y",
                "-i", str(audio_file),
                "-t", str(audio_duration),
                "-c:a", "aac",
                str(processed_audio)
            ]
            subprocess.run(trim_command, check=True)
        else:
            processed_audio = audio_file

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
            "-i", str(processed_audio),
            "-c:v", "copy",
            "-c:a", "aac",
            "-strict", "experimental",
            str(audio_video_output)
        ]
        subprocess.run(add_audio, check=True)

        # Clean up intermediate files
        combined_video.unlink()
        processed_audio.unlink()
        temp_list_file.unlink()

        # Add to the list for final combination
        intermediate_videos.append(audio_video_output)

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

def parse_cnn_article(article_url):
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

def scrape_cnn_homepage(url, limit):
    """
    Scrape headlines and articles from CNN.
    
    Args:
        url (str): The base URL of CNN.
        limit (int): Number of articles to fetch.

    Returns:
        list of dict: List containing headline, article text, and URL.
    """
    logging.info(f"Scrape {url}")
    response = requests.get(url)
    
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
            base_url = "https://www.cnn.com"
            article_url = base_url + href

        try:
            article_data = parse_cnn_article(article_url)
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

def parse_config(config):
    config_settings = {}
    try:
        config.read('config.ini')
        config_settings['mode'] = config.getint('DEFAULT', 'mode')
        if config_settings['mode'] == 2:
            config_settings['tts_worker_pool_size'] = config.getint('MELO_TTS', 'tts_worker_pool_size')
            config_settings['number_of_articles'] = config.getint('NEWS_VIDEOS', 'number_of_articles')
        config_settings['hf_home'] = config.get('DEFAULT', 'hf_home')
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
    def __init__(self, token):
        self.worker_id = str(uuid.uuid4())
        logging.info(f"Worker ID: {self.worker_id}")
        self.manager_url = "http://ec2-54-88-53-193.compute-1.amazonaws.com:8080"
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

def create_news_videos(config_settings):
    import ai_interfaces.hunyuan_video
    import ai_interfaces.meloTTS
    urls_to_make_videos_from = [["US", "https://www.cnn.com/us"],
                                    ["World", "https://www.cnn.com/world"],
                                    ["Politics", "https://www.cnn.com/politics"],
                                    ["Sports", "https://www.cnn.com/sport"],
                                    ["Entertainment", "https://www.cnn.com/entertainment"]]
    for video_type, url in urls_to_make_videos_from:
        create_news_video(video_type, url, config_settings)

def create_news_video(video_type, url, config_settings):
    try:
        # {headline, text, url}
        articles = scrape_cnn_homepage(url, config_settings["number_of_articles"])
    except Exception as e:
        logging.critical(f"Failed to scrape website - {e}")
        return
    if not articles:
        logging.critical("No articles scraped")
        return

    # TODO this is shit
    # generate summarized version of each article
    article_texts = [a["text"] for a in articles]
    article_summary_prompt_info = PromptInfo(PromptInfo.article_summary, article_texts)
    for i, summary in enumerate(generate_text(article_summary_prompt_info, config_settings)):
        articles[i]["summary"] = summary

    # turn article summaries into audio
    try:
        articles_id_and_summary = [(article["id"], article["summary"]) for article in articles]
        melo_tts_multithread(articles_id_and_summary , config_settings["tts_worker_pool_size"])
    except Exception as e:
        logging.critical(f"Failed to generate audio - {e}")
        logging.info(f"Article data: {articles}")
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
        return

    # Generate videos. Will generate multiple <video_duration> second videos 
    # for each audio clip to cover the entire clip and more. Output file format is <audio_file_id>_<video_file_id>.mp4
    audio_dir = "tmp/audio"
    audio_file_durations = get_wav_duration(audio_dir)
    video_generation_data = {}
    video_duration = 4
    for a in articles:
        id_s = str(a["id"])
        video_generation_data[id_s] = []
        num_videos = math.ceil(audio_file_durations[id_s]/video_duration)
        for sub_id in range(num_videos):
            # for the last video, make it so the combined durations of all 
            # videos is audio_duration + 1
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
        audio_to_video_files = generate_videos_hunyuan(video_generation_data)
    except Exception as e:
        logging.critical(f"Failed to generate videos - {e}")
        logging.info(f"Video generation data: {video_generation_data}")
        return

    # TODO add fade between each grouping of videos
    # TODO generate different prompt for each video instead of multiple videos from the same prompt
    # TODO add background music
    # TODO add subtitles for narration
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_file_name = f"finished_video_{time_stamp}"
    #audio_to_video_files = {"0": {"video_files: ["0_0", "0_1", "0_2", "0_3"]}, "1": {"video_files": ["1_0", "1_1", "1_2", "1_3"]}}
    try:
        process_videos_and_audio(audio_to_video_files, output_file_name)
    except Exception as e:
        logging.critical(f"Failed to process videos and audio - {e}")
        logging.info(f"Audio and video files: {audio_to_video_files}")
        return

    summaries_concatinated = [" ".join(a["summary"] for a in articles)]
    title_prompt_info = PromptInfo(PromptInfo.make_title, summaries_concatinated)
    generated_title = generate_text(title_prompt_info, config_settings)
    # max title length for youtube is 100 chars
    generated_title_cut = generated_title[0][:75]
    # TODO use https://github.com/makiisthenes/TiktokAutoUploader to copy youtube video to tiktok
    title = f"{video_type} News - {generated_title_cut}"
    try:
        upload_to_youtube("tmp/"+output_file_name+".mp4", title)
    except Exception as e:
        logging.critical(f"Failed to upload to youtube - {e}")
        return

def create_topic_based_videos(config_settings, hf_token):
    from ai_interfaces.diffsynth import diffsynth_wan_multithread
    from ai_interfaces.stable_audio import generate_audio

    # generate videos
    os.environ['HF_HOME'] = config_settings["hf_home"]
    from huggingface_hub import login
    login(token=hf_token)
    with open("mode_0_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    # load one set of prompts from config based on day since start day
    day_since_start = (datetime.now() - datetime(2025, 3, 23)).days
    prompts_today = config["prompts"][day_since_start]
    logging.info(f"Generating videos and audio using the following prompts: {prompts_today}")
    logging.info("Generating videos")
    fps = 24
    video_duration = 5
    num_frames = fps * video_duration + 1
    try:
        # TODO fix filename, right now it's hardcoded to 0_X.wav
        diffsynth_wan_multithread(prompts_today["videos"], num_frames, fps)
    except Exception as e:
        logging.error(f"Exception while generating video: {e}")

    logging.info("Generating audio")
    # note - max duration of music is 47 seconds
    audio_duration = video_duration * len(prompts_today["videos"])
    audio_parameters = [{
        "prompt": f"{prompts_today['music']}",
        "seconds_start": 0, 
        "seconds_total": audio_duration
    }]
    try:
        # TODO fix filename, right now it's hardcoded to 0.wav
        generate_audio(audio_parameters)
    except Exception as e:
        logging.error(f"Exception while generating video: {e}")

    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_file_name = f"finished_video_{time_stamp}"
    audio_to_video_files = {"0": {"audio_duration": audio_duration, "video_files": ["0_0", "0_1", "0_2"]}}
    try:
        process_videos_and_audio(audio_to_video_files, output_file_name)
    except Exception as e:
        logging.critical(f"Failed to process videos and audio - {e}")
        logging.info(f"Audio and video files: {audio_to_video_files}")
        return

    # upload to yt, tiktok, and instagram
    # TODO tiktok, instagram
    summaries_concatinated = [" ".join(p for p in prompts_today["videos"])]
    title_prompt_info = PromptInfo(PromptInfo.make_title, summaries_concatinated)
    generated_title = generate_text(title_prompt_info, config_settings)
    # max title length for youtube is 100 chars
    generated_title_cut = generated_title[0][:99]
    # TODO use https://github.com/makiisthenes/TiktokAutoUploader to copy youtube video to tiktok
    try:
        upload_to_youtube("tmp/"+output_file_name+".mp4", generated_title_cut)
    except Exception as e:
        logging.critical(f"Failed to upload to youtube - {e}")
        return

def create_quote_based_videos(config_settings):
    from ai_interfaces.diffsynth import diffsynth_wan_multithread
    from ai_interfaces.stable_audio import generate_audio
    from ai_interfaces.kokoro_tts import text_to_speech

    logging.critical("Not implemented")
    return

def main():
    # TODO args cannot be used because they are picked up by the hunyuan video parser, and then an error will occur complaining about unrecognized args
    parser = argparse.ArgumentParser(description="Scrape news stories from CNN")
    parser.add_argument('--log', type=str, default='info', help="Log level (debug, info, warning, error, critical)")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config_settings = parse_config(config)
    if not config_settings or not args:
        logging.critical("Config or args error")
        return

    with open("tokens.yaml", "r") as file:
        tokens = yaml.safe_load(file)["tokens"]

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
    #manager_client = ManagerClient(tokens["big_kahuna"])
    #manager_client.register_with_manager()

    mode = config_settings["mode"]
    if mode == 0:
        create_topic_based_videos(config_settings, tokens["huggingface"])
    elif mode == 1:
        create_quote_based_videos(config_settings)
    elif mode == 2:
        create_news_videos(config_settings)

    # TODO add cleanup for logs and finished video files
    #manager_client.notify_task_completed()

if __name__ == "__main__":
    main()
