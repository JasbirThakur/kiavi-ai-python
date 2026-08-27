import requests
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import wikipediaapi
from config import BRANDFETCH_API_KEY

wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent='KiaviIQ/1.0 (contact@appdeft.ai)'
)

def fetch_brand_logo(domain: str) -> str:
    """Fetches high-res brand logo using Brandfetch API with Clearbit fallback"""
    clean_domain = domain.replace("https://", "").replace("http://", "").split("/")[0].strip()
    
    if BRANDFETCH_API_KEY:
        try:
            url = f"https://api.brandfetch.io/v2/brands/{clean_domain}"
            headers = {"Authorization": f"Bearer {BRANDFETCH_API_KEY}"}
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                for logo in data.get("logos", []):
                    for fmt in logo.get("formats", []):
                        if fmt.get("src"):
                            return fmt["src"]
        except Exception as e:
            print(f"Brandfetch Error: {e}")
            
    # Free Fallback to Google High-Res Favicon CDN
    return f"https://www.google.com/s2/favicons?domain={clean_domain}&sz=128"

def scrape_wikipedia_topic(topic_or_url: str) -> tuple[str, str]:
    """Extracts rich prose from Wikipedia via API"""
    clean_topic = topic_or_url.split('/wiki/')[-1].replace('_', ' ').strip()
    page = wiki.page(clean_topic)
    
    if not page.exists():
        raise Exception(f"Wikipedia page not found for topic: {clean_topic}")
        
    return f"Wikipedia: {page.title}", page.text

def scrape_url_content(url: str) -> tuple[str, str, str]:
    """
    Intelligent Web Scraper:
    - Detects Wikipedia links and uses official Wikipedia API
    - Uses BeautifulSoup for standard web domains
    - Returns: (Title, Clean Prose Content, Logo URL)
    """
    if "wikipedia.org/wiki/" in url:
        title, text = scrape_wikipedia_topic(url)
        logo = "https://en.wikipedia.org/static/favicon/wikipedia.ico"
        return title, text, logo

    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        
    domain = urlparse(url).netloc
    logo_url = fetch_brand_logo(domain)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string.strip() if soup.title and soup.title.string else domain
        
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'button', 'form']):
            tag.decompose()
            
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = '\n'.join(lines)
        
        return title, clean_text, logo_url

    except Exception as e:
        raise Exception(f"Failed to scrape website: {str(e)}")