import requests
from bs4 import BeautifulSoup
import logging
import argparse

def scrape(base_url, limit=5):
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
        if href not in seen_hrefs:
            seen_hrefs.add(href)
            article_url = base_url + href
            
            try:
                # Fetch the article
                article_response = requests.get(article_url)
                if article_response.status_code != 200:
                    logging.warning(f"Failed to fetch article at {article_url}")
                    continue
                
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                
                # Extract headline
                headline = article_soup.find('h1')
                if not headline:
                    logging.info(f"No headline found for {article_url}")
                    continue
                
                # Extract article text
                paragraphs = article_soup.find_all('div', class_='paragraph')
                article_text = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                # TODO fix stupid AI code
                print(article_text)
                
                if article_text:
                    articles.append({
                        'headline': headline.get_text(strip=True),
                        'text': article_text,
                        'url': article_url
                    })
                    logging.info(f"Successfully scraped article: {headline.get_text(strip=True)}")
                
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
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    
    # Scrape CNN
    base_url = "https://www.cnn.com"
    articles = scrape(base_url, limit=args.limit)
    
    # Display articles
    for article in articles:
        print(f"Headline: {article['headline']}\n")
        print(f"Text: {article['text'][:500]}...\n")  # Print the first 500 characters
        print(f"URL: {article['url']}\n")
        print("=" * 80)
