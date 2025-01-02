import requests
from bs4 import BeautifulSoup
import logging
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import os

from melo.api import TTS

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler


class prompt_info:
    article_summary = "article_summary"
    video_prompt = "video_prompt"
    def __init__(self, prompt_type, prompt_data):
        self.prompts = prompt_data
        if prompt_type == self.article_summary:
            self.system_prompt = "Summarize the input article into 2 sentences. You cannot write anything except for the article summary. Do not write something like 'Here is a summary of the article:', you can only write the summary."
            self.prompt_type = self.article_summary
        elif prompt_type == video_prompt:
            self.system_prompt = "Write a 1 sentence prompt for a text to video LLM that would generate a video for the following article. Do not write anything except for the prompt. Do not include the time duration of the video. Here is an example prompt: 'A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.'"
            self.prompt_type = video_prompt

def generate_videos(prompts):
    # this has to be done, some of the hunyuan scripts have hardcoded paths that expect the cwd to be hunyuan
    os.chdir("HunyuanVideo")

    args = parse_args()
    args.prompt = "Stocks just did something they haven't done in nearly three decades"
    args.video_size = (960, 544)
    # determine video length from length of voiceover. Framerate is 24
    args.video_length = 129
    args.seed = 0
    args.infer_steps = 30
    args.use_cpu_offload = True
    args.save_path = "./results"

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

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
            time_flag = datetime.now().strftime("%Y_%m_%d")
            save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f'Sample save to: {save_path}')

    os.chdir("..")

def process_videos_and_audio(audio_video_mapping, final_output_file):
    # TODO this needs to be fixed up and tested
    """
    Combines video files associated with audio files, adds the audio, 
    and combines all the resulting videos into one.

    Args:
        audio_video_mapping (dict): Mapping where keys are audio file paths 
                                    and values are lists of associated video file paths.
        final_output_file (str): Path to the final combined video file.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    try:
        intermediate_videos = []

        # Process each audio and associated video files
        for audio_file, video_files in audio_video_mapping.items():
            audio_file = Path(audio_file)
            video_files = [Path(video) for video in video_files]
            
            # Check if all files exist
            for file in [audio_file] + video_files:
                if not file.exists():
                    print(f"Error: File not found: {file}")
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
            str(final_output_file)
        ]
        subprocess.run(combine_all_command, check=True)

        # Clean up intermediate videos and the final list file
        for video in intermediate_videos:
            video.unlink()
        final_list_file.unlink()

        print(f"Final combined video saved to {final_output_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg execution: {e}")
        return False

def generate_text(prompt_info):
    repo_path = Path(__file__).parent.as_posix()
    llamaCpp_file_path = repo_path + "/llama.cpp/build/bin/llama-cli"
    model_file_path = repo_path + "/models/Llama-3.1-8B-Lexi-Uncensored-V2-Q6_K_L.gguf"
    cpu_threads = "8"
    output_len = "64"
    # TODO change this to match machine
    gpu_layers = "12"
    output_len = "150"
    context_len = "10000"

    outputs = []

    full_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Cutting Knowledge Date: December 2023
    Today Date: 26 Jul 2024

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    for prompt in prompt_info.prompts:
        complete_prompt = full_prompt.replace("{system_prompt}", prompt_info.system_prompt)
        complete_prompt = complete_prompt.replace("{prompt}", prompt)
        try:
            result = subprocess.run(
                [llamaCpp_file_path,
                 "-m", model_file_path,
                 "-t", cpu_threads,
                 #"-n", output_len,
                 "-ngl", gpu_layers,
                 "--temp", "0.9",
                 "-c", context_len,
                 "-p", complete_prompt],
                capture_output=True,
                text=True
            )
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
    print("outputs =", outputs)
    return outputs

def tts(article_data):
    # TODO make this a for loop to generate speech for summarized version
    speed = 1.2
    device = 'auto' # Will automatically use GPU if available
    model = TTS(language='EN', device=device)
    speaker_ids = model.hps.data.spk2id
    output_path = 'en-us.wav'
    # TODO replace first arg
    model.tts_to_file(article_data['headline'], speaker_ids['EN-US'], output_path, speed=speed)

def parse_article(article_url):
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
    logging.info("Starting scrape for CNN...")
    response = requests.get(base_url)
    
    if response.status_code != 200:
        logging.error(f"Failed to retrieve CNN homepage. Status code: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = []
    
    seen_hrefs = set()
    # Find article links matching the pattern
    for link in soup.find_all('a', "container__link container__link--type-article container_lead-package__link container_lead-package__left container_lead-package__light", href=True):
        # TODO add more classes. This class is only for the stories with a picture (?)
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
            if len(articles) >= limit:
                break
        except Exception as e:
            logging.error(f"Error while processing article at {article_url}: {e}")
                
    return articles

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Scrape news stories from CNN")
    parser.add_argument('--limit', type=int, default=5, help="Number of articles to scrape")
    parser.add_argument('--log', type=str, default='info', help="Log level (debug, info, warning, error, critical)")
    args = parser.parse_args()

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
    articles = scrape_homepage(base_url, limit=args.limit)

    # TODO this is shit
    # generate summarized version of each article
    article_texts = [a["text"] for a in articles]
    article_summary_prompt_info = prompt_info(
                                    prompt_info.article_summary, 
                                    article_texts)
    for i, summary in enumerate(generate_text(article_summary_prompt_info)):
        articles[i]["summary"] = summary

    # TODO turn article summaries into audio
    tts(articles)

    # TODO generate prompt for text to video LLM using summarized articles
    summarized_articles = [a[prompt_info.article_summary] for a in article_summary_prompt_info] 
    video_prompt_prompt_info = prompt_info(
                                prompt_info.video_prompt,
                                summarized_articles)
    articles["video_prompt"] = generate_text(video_prompt_prompt_info)

