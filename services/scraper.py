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

from urllib.parse import urljoin

def scrape_single_page(url: str, headers: dict) -> tuple[str, str, list[str]]:
    """Scrapes a single page, preserving footer, contact info, headers, and image alt descriptions."""
    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        
        # Collect internal links before cleaning
        domain = urlparse(url).netloc
        internal_links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(url, href)
            p = urlparse(full_url)
            if p.netloc == domain and not any(full_url.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.svg', '.pdf', '.css', '.js']):
                clean_link = full_url.split('#')[0].rstrip('/')
                if clean_link and clean_link not in internal_links and clean_link != url.rstrip('/'):
                    internal_links.append(clean_link)

        # Convert image alt attributes into readable text so diagram and visual information is indexed
        for img in soup.find_all('img', alt=True):
            alt_text = img['alt'].strip()
            if alt_text and len(alt_text) > 3:
                img.replace_with(soup.new_string(f" [Visual Plate / Image: {alt_text}] "))

        # Decompose only non-content executable/styling elements (NEVER decompose footer, nav, header, address)
        for tag in soup(['script', 'style', 'noscript', 'svg', 'iframe']):
            tag.decompose()
            
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = '\n'.join(lines)
        
        return title, clean_text, internal_links
    except Exception as e:
        print(f"[Scrape Page Warning] {url}: {e}")
        return "", "", []

def scrape_url_content(url: str, crawl_depth: int = 6) -> tuple[str, str, str]:
    """
    Intelligent Deep Web Scraper:
    - Detects Wikipedia links and uses official Wikipedia API
    - Crawls primary landing page + key subpages (/about, /contact, /services, /case-studies)
    - Preserves contact details, phone numbers, addresses, footers, and diagrams
    - Returns: (Title, Combined Prose Knowledge, Logo URL)
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
        main_title, main_text, found_links = scrape_single_page(url, headers)
        if not main_text:
            raise Exception("No readable text found on the target website.")

        collected_pages = [(url, main_title, main_text)]
        
        # Prioritize key business subpages
        priority_keywords = ['about', 'contact', 'service', 'solution', 'pricing', 'case-stud', 'team', 'company', 'developer']
        prioritized_links = []
        for kw in priority_keywords:
            for l in found_links:
                if kw in l.lower() and l not in prioritized_links and l != url:
                    prioritized_links.append(l)

        # Append remaining links
        for l in found_links:
            if l not in prioritized_links and l != url:
                prioritized_links.append(l)

        # Scrape top subpages up to crawl_depth limit
        crawled_count = 0
        for sub_url in prioritized_links[:crawl_depth]:
            s_title, s_text, _ = scrape_single_page(sub_url, headers)
            if s_text and len(s_text) > 100:
                collected_pages.append((sub_url, s_title, s_text))
                crawled_count += 1

        print(f"🕸️ [Web Crawler]: Scraped {len(collected_pages)} pages for domain {domain}")

        # Combine all pages into structured knowledge blocks
        formatted_sections = []
        for p_url, p_title, p_content in collected_pages:
            formatted_sections.append(f"=== Web Page: {p_title} ({p_url}) ===\n{p_content}")

        combined_text = "\n\n" + ("\n" + "="*50 + "\n\n").join(formatted_sections)
        return main_title, combined_text, logo_url

    except Exception as e:
        raise Exception(f"Failed to scrape website: {str(e)}")