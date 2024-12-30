import requests
from bs4 import BeautifulSoup
import logging
import argparse
import datetime
import pathlib
import subprocess

def llm_summarize_articles(articles):
    repo_path = pathlib.Path(__file__).parent.as_posix()
    llamaCpp_file_path = repo_path + "/llama.cpp/build/bin/llama-cli"
    model_file_path = repo_path + "/models/Llama-3.1-8B-Lexi-Uncensored-V2-Q6_K_L.gguf"
    cpu_threads = "8"
    output_len = "64"
    gpu_layers = "12"
    output_len = "150"
    context_len = "10000"

    for article in articles:
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        prompt += "\n\n### Instruction:\n" + "Please summarize the following article into two sentences. You CANNOT write anything else except the article summary." + "\n\n### Input: " + article['text'] + "\n\n### Response:\n"
        try:
            result = subprocess.run(
                [llamaCpp_file_path,
                 "-m", model_file_path,
                 "-t", cpu_threads,
                 "-n", output_len,
                 "-ngl", gpu_layers,
                 "--temp", "0.9",
                 "-c", context_len,
                 "-p", prompt],
                capture_output=True,
                text=True
            )
            llm_output = result.stdout.strip()
            start_string = "Response:\n"
            start_idx = llm_output.find(start_string) + len(start_string)
            extra_ending = " [end of text]"
            llm_output_stripped = llm_output[start_idx:-len(extra_ending)]
            article["summarized"] = llm_output_stripped
        except Exception as e:
            logging.error(f"Error running subprocess: {e}")
            continue

def tts(article_data):
    from melo.api import TTS

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
    year_month_date = datetime.datetime.now().strftime("%Y_%m_%d")
    logging.basicConfig(
        level=log_level,
		filename = 'rooporter_' + year_month_date + '.log',
        format='%(asctime)s - %(levelname)s - %(message)s',
		datefmt='%H:%M:%S'
    )

    base_url = "https://www.cnn.com"
    articles = scrape_homepage(base_url, limit=args.limit)

    llm_summarize_articles(articles)

    # TODO run TTS on summarized article text
    #tts(article_data)
