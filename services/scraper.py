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
import subprocess
import shutil

CHROME_BIN = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")

def fetch_html_content(url: str, headers: dict) -> str:
    """
    Fetches raw or dynamically rendered HTML.
    If headless Chrome is available, renders client-side JS (SPAs, React, Next.js, Vue).
    Falls back gracefully to requests.get().
    """
    if CHROME_BIN:
        try:
            cmd = [
                CHROME_BIN,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--dump-dom",
                "--timeout=12000",
                url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and len(result.stdout) > 500:
                return result.stdout
        except Exception as e:
            print(f"⚠️ [Headless Chrome Fallback] {url}: {e}")

    resp = requests.get(url, headers=headers, timeout=12)
    resp.raise_for_status()
    return resp.text

def scrape_single_page(url: str, headers: dict) -> tuple[str, str, list[str], dict]:
    """Scrapes a single page, preserving social links, contact info, headers, meta tags, and image descriptions."""
    try:
        raw_html = fetch_html_content(url, headers)
        soup = BeautifulSoup(raw_html, 'html.parser')
        title = soup.title.string.strip() if soup.title and soup.title.string else url

        # 1. Extract Meta Description and OpenGraph metadata
        meta_desc = ""
        desc_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if desc_tag and desc_tag.get('content'):
            meta_desc = desc_tag['content'].strip()

        # 2. Extract Social Media Channels and Contact Links
        domain = urlparse(url).netloc
        social_links = {}
        internal_links = []

        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            full_url = urljoin(url, href)
            p = urlparse(full_url)
            lower_href = full_url.lower()

            # Social channels detection
            if 'linkedin.com' in lower_href:
                social_links['LinkedIn'] = full_url
                a.replace_with(soup.new_string(f" [Social Link: LinkedIn ({full_url})] "))
            elif 'instagram.com' in lower_href:
                social_links['Instagram'] = full_url
                a.replace_with(soup.new_string(f" [Social Link: Instagram ({full_url})] "))
            elif 'twitter.com' in lower_href or 'x.com' in lower_href:
                social_links['X/Twitter'] = full_url
                a.replace_with(soup.new_string(f" [Social Link: X/Twitter ({full_url})] "))
            elif 'github.com' in lower_href and domain not in lower_href:
                social_links['GitHub'] = full_url
                a.replace_with(soup.new_string(f" [Social Link: GitHub ({full_url})] "))
            elif 'youtube.com' in lower_href:
                social_links['YouTube'] = full_url
                a.replace_with(soup.new_string(f" [Social Link: YouTube ({full_url})] "))
            elif 'facebook.com' in lower_href:
                social_links['Facebook'] = full_url
                a.replace_with(soup.new_string(f" [Social Link: Facebook ({full_url})] "))
            elif href.startswith('mailto:'):
                email = href.replace('mailto:', '').split('?')[0].strip()
                a.replace_with(soup.new_string(f" [Contact Email: {email}] "))
            elif href.startswith('tel:'):
                phone = href.replace('tel:', '').strip()
                a.replace_with(soup.new_string(f" [Contact Phone: {phone}] "))
            elif p.netloc == domain and not any(full_url.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.svg', '.pdf', '.css', '.js']):
                clean_link = full_url.split('#')[0].rstrip('/')
                if clean_link and clean_link not in internal_links and clean_link != url.rstrip('/'):
                    internal_links.append(clean_link)

        # 3. Convert image alt attributes into readable text for visual grounding
        for img in soup.find_all('img', alt=True):
            alt_text = img['alt'].strip()
            if alt_text and len(alt_text) > 3:
                img.replace_with(soup.new_string(f" [Visual Plate / Image: {alt_text}] "))

        # 4. Decompose non-content executable tags
        for tag in soup(['script', 'style', 'noscript', 'svg', 'iframe']):
            tag.decompose()

        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = '\n'.join(lines)

        metadata = {
            'meta_desc': meta_desc,
            'social_links': social_links
        }
        return title, clean_text, internal_links, metadata
    except Exception as e:
        print(f"[Scrape Page Warning] {url}: {e}")
        return "", "", [], {}

def scrape_url_content(url: str, crawl_depth: int = 6) -> tuple[str, str, str]:
    """
    Intelligent Deep Web Scraper:
    - Extracts complete metadata, social links (LinkedIn, X/Twitter, Instagram, GitHub), contact channels
    - Crawls primary landing page + key subpages
    - Filters out broken/empty stubs
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
        main_title, main_text, found_links, main_meta = scrape_single_page(url, headers)
        if not main_text:
            raise Exception("No readable text found on the target website.")

        collected_pages = [(url, main_title, main_text)]
        all_socials = dict(main_meta.get('social_links', {}))

        # Prioritize key business subpages
        priority_keywords = ['about', 'contact', 'service', 'solution', 'pricing', 'case-stud', 'team', 'company', 'developer', 'faq', 'feature']
        prioritized_links = []
        for kw in priority_keywords:
            for l in found_links:
                if kw in l.lower() and l not in prioritized_links and l != url:
                    prioritized_links.append(l)

        for l in found_links:
            if l not in prioritized_links and l != url:
                prioritized_links.append(l)

        # Scrape top subpages up to crawl_depth limit
        for sub_url in prioritized_links[:crawl_depth]:
            s_title, s_text, _, s_meta = scrape_single_page(sub_url, headers)
            # Skip empty or stub pages
            if s_text and len(s_text) > 120 and "no content available at this time" not in s_text.lower():
                collected_pages.append((sub_url, s_title, s_text))
                if s_meta.get('social_links'):
                    all_socials.update(s_meta['social_links'])

        print(f"🕸️ [Web Crawler]: Scraped {len(collected_pages)} pages for domain {domain} with {len(all_socials)} social channels.")

        # Structured Knowledge Summary Header
        overview_lines = [
            f"=== Organization & Website Profile: {main_title} ===",
            f"Official Website: {url}",
            f"Domain: {domain}"
        ]
        if main_meta.get('meta_desc'):
            overview_lines.append(f"Primary Mission & Meta Summary: {main_meta['meta_desc']}")

        if all_socials:
            overview_lines.append("Verified Social Media & Follow Links:")
            for platform, s_url in all_socials.items():
                overview_lines.append(f"- {platform}: {s_url}")

        overview_block = "\n".join(overview_lines)

        # Combine all pages into structured knowledge blocks
        formatted_sections = [overview_block]
        for p_url, p_title, p_content in collected_pages:
            formatted_sections.append(f"=== Web Page: {p_title} ({p_url}) ===\n{p_content}")

        combined_text = "\n\n" + ("\n" + "="*50 + "\n\n").join(formatted_sections)
        return main_title, combined_text, logo_url

    except Exception as e:
        raise Exception(f"Failed to scrape website: {str(e)}")
