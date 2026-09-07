import requests
import re
import json
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import wikipediaapi
import subprocess
import shutil
from config import BRANDFETCH_API_KEY

wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent='KiaviIQ/1.0 (contact@appdeft.ai)'
)

def clean_domain_name(domain: str) -> str:
    """Normalizes domains by stripping protocols, www, and paths."""
    d = domain.replace("https://", "").replace("http://", "").split("/")[0].strip()
    if d.startswith("www."):
        d = d[4:]
    return d.lower()

def is_valid_email(email: str) -> bool:
    """Validates email addresses and filters out CSS params, font weights, and asset filenames."""
    if not email or not isinstance(email, str):
        return False
    email = email.strip()
    if len(email) < 6 or len(email) > 100:
        return False
    lower = email.lower()
    bad_tokens = [
        "wght@", "opsz@", "@2x", "@3x", "..", ".woff", ".ttf",
        ".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif", ".js", ".css",
        "example.com", "sentry.io", "schema.org", "domain.com", "test.com"
    ]
    if any(b in lower for b in bad_tokens):
        return False
    pattern = r"^[a-zA-Z0-9_.+-]+@(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,12}$"
    return bool(re.match(pattern, email))

def is_valid_phone(phone: str) -> bool:
    """Validates phone numbers, ensuring acceptable digit count and no alphabetic code tokens."""
    if not phone or not isinstance(phone, str):
        return False
    phone = phone.strip()
    if len(phone) < 7 or len(phone) > 30:
        return False
    if re.search(r"[a-zA-Z]", phone):
        return False
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 7 or len(digits) > 15:
        return False
    if len(set(digits)) == 1:
        return False
    if digits in ("1234567890", "0123456789"):
        return False
    return True

def is_valid_address(addr: str) -> bool:
    """
    Validates physical addresses.
    Rejects base64 strings, font hashes, minified scripts, and programming tokens.
    Requires at least 3 words, digits, and a recognized address/geographic indicator.
    """
    if not addr or not isinstance(addr, str):
        return False
    clean = " ".join(addr.replace("\n", " ").replace("\r", " ").split()).strip()
    if len(clean) < 15 or len(clean) > 250:
        return False
    words = clean.split()
    if len(words) < 3:
        return False
    # Reject long hash/base64 tokens
    if any(len(w) > 24 for w in words):
        return False
    code_indicators = [
        "+", "=", "{", "}", "<", ">", "_id", "slug", "href", "http://", "https://",
        "__next", "function", "var ", "const ", "return", "typeof", "undefined",
        "null", "window.", "document.", "px;", "rem;", "rgba(", "rgb("
    ]
    lower = clean.lower()
    if any(tok in lower for tok in code_indicators):
        return False
    # Must contain at least one digit (building #, floor #, sector #, pin/zip code)
    if not re.search(r"\d", clean):
        return False
    addr_keywords = [
        "floor", "suite", "ste", "phase", "sector", "industrial", "road", "rd", "street", "st",
        "avenue", "ave", "boulevard", "blvd", "building", "bldg", "block", "plot", "sco",
        "plaza", "tower", "park", "way", "drive", "dr", "lane", "ln", "court", "ct", "circle",
        "cir", "highway", "hwy", "center", "centre", "square", "sq", "terrace",
        "city", "state", "county", "district", "province", "zip", "postal", "pincode", "box", "po box",
        "punjab", "chandigarh", "mohali", "delhi", "mumbai", "bangalore", "bengaluru",
        "hyderabad", "chennai", "kolkata", "pune", "noida", "gurugram", "gurgaon",
        "california", "texas", "florida", "new york", "washington", "san bruno",
        "mountain view", "san francisco", "san jose", "seattle", "austin", "chicago",
        "london", "uk", "united kingdom", "usa", "united states", "india", "canada", "australia"
    ]
    if not any(re.search(rf"\b{kw}\b", lower) for kw in addr_keywords):
        return False
    return True


def fetch_brand_logo(domain: str) -> str:
    """Fetches high-res brand logo using Brandfetch API with Google Favicon CDN fallback."""
    clean_domain = clean_domain_name(domain)

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
    """Extracts rich prose from Wikipedia via API."""
    clean_topic = topic_or_url.split('/wiki/')[-1].replace('_', ' ').strip()
    page = wiki.page(clean_topic)

    if not page.exists():
        raise Exception(f"Wikipedia page not found for topic: {clean_topic}")

    return f"Wikipedia: {page.title}", page.text

try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

CHROME_BIN = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")

def fetch_html_content(url: str, headers: dict) -> str:
    """Fetches raw or dynamically rendered HTML via curl_cffi (impersonating modern Chrome to bypass Cloudflare/Akamai), Headless Chrome, or requests."""
    # 1. High-Precision Browser Fingerprint Impersonation (bypasses Akamai Bot Manager, Cloudflare, WAF)
    if HAS_CURL_CFFI:
        try:
            resp = cffi_requests.get(
                url,
                impersonate="chrome124",
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "Upgrade-Insecure-Requests": "1"
                },
                timeout=18
            )
            if resp.status_code in (200, 201) and len(resp.text) > 300:
                return resp.text
            elif resp.status_code == 403:
                print(f"⚠️ [curl_cffi 403 notice]: {url}")
        except Exception as cffi_err:
            print(f"⚠️ [curl_cffi Impersonate Notice] {url}: {cffi_err}")

    # 2. Headless Chrome local execution if available
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

    # 3. Standard requests fallback
    resp = requests.get(url, headers=headers, timeout=12)
    resp.raise_for_status()
    return resp.text

def extract_modern_framework_data(html: str, base_url: str) -> dict:
    """
    Extracts structured content from Next.js (App Router RSC & Pages), Nuxt, React,
    and Schema.org JSON-LD microdata before script stripping.
    """
    extracted_text_blocks = []
    found_links = set()
    found_phones = set()
    found_emails = set()
    found_addresses = set()
    base_clean = clean_domain_name(urlparse(base_url).netloc)

    # 1. Next.js 13/14/15 App Router React Server Components (RSC) streams
    rsc_chunks = re.findall(r'self\.__next_f\.push\(\[1,\s*\"(.*?)\"\]\)', html)
    for c in rsc_chunks:
        try:
            decoded = c.encode().decode('unicode-escape', errors='ignore').replace('\x00', '').replace('\0', '').replace('\\u0000', '')
            # Extract URLs
            raw_urls = re.findall(r'\"(?:href|megaMenuTitleUrl|url|path|slug)\":\s*\"([^\"]+)\"', decoded)
            for u in raw_urls:
                if u.startswith('/') or base_clean in clean_domain_name(urlparse(u).netloc):
                    full = urljoin(base_url, u)
                    found_links.add(full.split('#')[0].rstrip('/'))

            # Extract phone numbers
            phones = re.findall(r'\"phone\":\s*\"([^\"]+)\"', decoded)
            for ph in phones:
                if is_valid_phone(ph):
                    found_phones.add(ph)

            # Extract addresses
            addrs = re.findall(r'\"address\":\s*\"([^\"]+)\"', decoded)
            for ad in addrs:
                if is_valid_address(ad):
                    found_addresses.add(ad)

            # Extract readable prose segments
            prose_matches = re.findall(r'\"(?:title|description|text|content|name|role|answer|question)\":\s*\"([^\"]{10,})\"', decoded)
            for p in prose_matches:
                p_clean = p.replace('\\n', ' ').strip()
                if p_clean and len(p_clean) > 12:
                    extracted_text_blocks.append(p_clean)
        except Exception:
            pass

    # 2. Next.js Pages Router (__NEXT_DATA__)
    next_data_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
    if next_data_match:
        try:
            data = json.loads(next_data_match.group(1))
            def extract_strings(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in ('address', 'location', 'headquarters') and isinstance(v, str):
                            if is_valid_address(v):
                                found_addresses.add(v)
                        elif k in ('phone', 'telephone', 'mobile') and isinstance(v, str):
                            if is_valid_phone(v):
                                found_phones.add(v)
                        elif k in ('email', 'contactEmail') and isinstance(v, str):
                            if is_valid_email(v):
                                found_emails.add(v)
                        elif isinstance(v, str) and len(v) > 20:
                            extracted_text_blocks.append(v)
                        else:
                            extract_strings(v)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_strings(item)
            extract_strings(data.get('props', {}))
        except Exception:
            pass

    # 3. Schema.org JSON-LD Microdata (Organization, LocalBusiness, PostalAddress)
    ld_json_matches = re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL)
    for ld in ld_json_matches:
        try:
            ld_obj = json.loads(ld.strip())
            if isinstance(ld_obj, list):
                items = ld_obj
            else:
                items = [ld_obj]
            for item in items:
                if item.get('telephone') and is_valid_phone(str(item['telephone'])):
                    found_phones.add(str(item['telephone']))
                if item.get('email') and is_valid_email(str(item['email'])):
                    found_emails.add(str(item['email']))
                addr = item.get('address')
                if isinstance(addr, dict):
                    parts = [
                        addr.get('streetAddress', ''),
                        addr.get('addressLocality', ''),
                        addr.get('addressRegion', ''),
                        addr.get('postalCode', ''),
                        addr.get('addressCountry', '')
                    ]
                    full_addr = ", ".join([str(p).strip() for p in parts if p]).strip()
                    if is_valid_address(full_addr):
                        found_addresses.add(full_addr)
                elif isinstance(addr, str) and is_valid_address(addr):
                    found_addresses.add(addr)
        except Exception:
            pass

    # 4. React / Redux Preloaded & Hydration State Parsing (window.__PRELOADED_STATE__, window.__USER_PRELOADED_STATE__, window.__INITIAL_STATE__, window.__INITIAL_DATA__, window.__APOLLO_STATE__, etc.)
    state_names = [
        "__PRELOADED_STATE__", "__USER_PRELOADED_STATE__", "__INITIAL_STATE__",
        "__INITIAL_DATA__", "__APOLLO_STATE__", "__APP_INITIAL_STATE__", "__SSR_DATA__"
    ]
    decoder = json.JSONDecoder()
    for sname in state_names:
        for match in re.finditer(rf"window\.{sname}\s*=\s*", html):
            start_idx = match.end()
            brace_idx = html.find("{", start_idx)
            if brace_idx != -1 and (brace_idx - start_idx) < 15:
                try:
                    state_data, _ = decoder.raw_decode(html[brace_idx:])
                    def traverse_state(obj, depth=0):
                        if depth > 9:
                            return
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                k_lower = k.lower()
                                if isinstance(v, str):
                                    v_clean = v.strip()
                                    if not v_clean:
                                        continue
                                    if k_lower in ("telephone", "phone", "mobile", "helpline", "customercare", "contactnumber", "phone_number"):
                                        if is_valid_phone(v_clean):
                                            found_phones.add(v_clean)
                                    elif k_lower in ("email", "contactemail", "supportemail", "helpemail"):
                                        if is_valid_email(v_clean):
                                            found_emails.add(v_clean)
                                    elif k_lower in ("address", "headquarters", "officelocation", "streetaddress", "registered_office"):
                                        if is_valid_address(v_clean):
                                            found_addresses.add(v_clean)
                                    elif k_lower in ("url", "href", "slug", "link", "megamenutitleurl", "path") and (v_clean.startswith("/") or base_clean in clean_domain_name(urlparse(v_clean).netloc)):
                                        full = urljoin(base_url, v_clean)
                                        found_links.add(full.split('#')[0].rstrip('/'))
                                    elif 15 <= len(v_clean) <= 600 and not any(ch in v_clean for ch in ["{", "}", "<", ">", ";", "function(", "return ", "typeof "]):
                                        if any(indicator in k_lower for indicator in ("title", "desc", "name", "headline", "summary", "text", "content", "about", "query", "question", "answer", "category", "offer", "feature", "brand")):
                                            extracted_text_blocks.append(f"{k}: {v_clean}")
                                elif isinstance(v, (dict, list)):
                                    traverse_state(v, depth + 1)
                        elif isinstance(obj, list):
                            for item in obj[:80]:
                                traverse_state(item, depth + 1)
                    traverse_state(state_data)
                except Exception:
                    pass

    # 5. Regex Scanning for Global Phones, Emails, and Addresses in the Raw HTML
    raw_emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', html)
    for em in raw_emails:
        if is_valid_email(em):
            found_emails.add(em)

    # Physical Address Pattern Scanning
    for a in re.findall(r'address[^:]{0,5}:[^a-zA-Z0-9]{0,5}([^\"\\\}]{10,140})', html, re.I):
        clean_a = a.strip().replace('\\n', ', ').replace('\\', '').replace('\x00', '').strip()
        if is_valid_address(clean_a):
            found_addresses.add(clean_a)

    addr_patterns = [
        r'(?:Floor|Suite|Phase|Sector|Industrial Area|Block|Plot|SCO|Road|Street|Avenue|Boulevard|Building)[^<>\n\r\"\'\}]{10,100}(?:Punjab|Chandigarh|Mohali|Delhi|Mumbai|Bangalore|California|London|UK|USA|India|\d{5,6})'
    ]
    for pat in addr_patterns:
        for m in re.findall(pat, html, re.I):
            clean_m = m.strip().replace('\\n', ', ').replace('\\', '').replace('\x00', '').strip()
            if is_valid_address(clean_m):
                found_addresses.add(clean_m)

    return {
        "text_blocks": extracted_text_blocks,
        "links": list(found_links),
        "phones": list(found_phones),
        "emails": list(found_emails),
        "addresses": list(found_addresses)
    }

def scrape_single_page(url: str, headers: dict) -> tuple[str, str, list[str], dict]:
    """
    Scrapes a single web page, preserving headers, footers, sidebars, dropdowns,
    Next.js/React state, contact info, physical addresses, and image descriptions.
    """
    try:
        raw_html = fetch_html_content(url, headers)
        framework_data = extract_modern_framework_data(raw_html, url)
        soup = BeautifulSoup(raw_html, 'html.parser')
        title = soup.title.string.strip() if soup.title and soup.title.string else url

        domain = clean_domain_name(urlparse(url).netloc)
        social_links = {}
        internal_links = set(framework_data["links"])
        document_links = []

        # 1. Meta Description and OpenGraph
        meta_desc = ""
        desc_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if desc_tag and desc_tag.get('content'):
            meta_desc = desc_tag['content'].strip()

        # 2. Extract Header, Footer, and Address tags explicitly
        header_text = ""
        header_tag = soup.find('header')
        if header_tag:
            header_text = header_tag.get_text(separator=' ').strip()

        footer_text = ""
        footer_tag = soup.find('footer')
        if footer_tag:
            footer_text = footer_tag.get_text(separator=' ').strip()
            for line in footer_text.splitlines():
                line_clean = line.strip()
                if is_valid_address(line_clean):
                    framework_data["addresses"].append(line_clean)

        # Explicit <address> tag
        for addr_tag in soup.find_all('address'):
            t = addr_tag.get_text(separator=' ').strip()
            if is_valid_address(t):
                framework_data["addresses"].append(t)

        # 3. Extract Links, Emails, Phones, and Linked Documents
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            full_url = urljoin(url, href)
            p = urlparse(full_url)
            lower_href = full_url.lower()

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
                if is_valid_email(email):
                    framework_data["emails"].append(email)
                    a.replace_with(soup.new_string(f" [Contact Email: {email}] "))
            elif href.startswith('tel:'):
                phone = href.replace('tel:', '').strip()
                if is_valid_phone(phone):
                    framework_data["phones"].append(phone)
                    a.replace_with(soup.new_string(f" [Contact Phone: {phone}] "))
            else:
                # Check for linked PDFs, CSVs, or brochures
                if any(lower_href.endswith(ext) for ext in ['.pdf', '.csv']):
                    document_links.append(full_url)
                # Internal Page Link
                elif clean_domain_name(p.netloc) == domain and not any(full_url.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.svg', '.css', '.js', '.webp', '.ico']):
                    clean_link = full_url.split('#')[0].rstrip('/')
                    if clean_link and clean_link != url.rstrip('/'):
                        internal_links.add(clean_link)

        # 4. Convert image alt attributes into readable text for visual grounding
        for img in soup.find_all('img', alt=True):
            alt_text = img['alt'].strip()
            if alt_text and len(alt_text) > 3:
                img.replace_with(soup.new_string(f" [Visual Plate / Image: {alt_text}] "))

        # 5. Decompose non-content executable script/style tags
        for tag in soup(['script', 'style', 'noscript', 'svg', 'iframe']):
            tag.decompose()

        # 5.1 Extract physical address candidates from DOM leaf blocks (p, div, li, span, section, address)
        for block in soup.find_all(['p', 'div', 'li', 'span', 'section', 'address']):
            if block.find(['p', 'div', 'section']):
                continue
            block_lines = [l.strip().rstrip(',') for l in block.get_text(separator='\n').splitlines() if l.strip()]
            if block_lines:
                candidate = ", ".join(block_lines)
                candidate = re.sub(r'\s+', ' ', candidate)
                candidate = re.sub(r',\s*,+', ', ', candidate).strip(' ,')
                if is_valid_address(candidate):
                    framework_data["addresses"].append(candidate)

        # 6. Assemble all prose text
        dom_text = soup.get_text(separator='\n')
        lines = [line.strip() for line in dom_text.splitlines() if line.strip()]

        # Append framework data (Next.js RSC, decoded props, Schema.org)
        if framework_data["text_blocks"]:
            lines.append("\n=== Modern Framework & Client Component Data ===")
            lines.extend(framework_data["text_blocks"])

        clean_text = '\n'.join(lines)

        metadata = {
            'meta_desc': meta_desc,
            'social_links': social_links,
            'phones': list(set(framework_data["phones"])),
            'emails': list(set(framework_data["emails"])),
            'addresses': list(set(framework_data["addresses"])),
            'document_links': document_links,
            'header_text': header_text,
            'footer_text': footer_text
        }
        return title, clean_text, list(internal_links), metadata
    except Exception as e:
        print(f"[Scrape Page Warning] {url}: {e}")
        return "", "", [], {}

def discover_sitemap_urls(base_url: str, headers: dict) -> list[str]:
    """Probes /sitemap.xml and robots.txt to discover all subpages across any modern website framework."""
    discovered = []
    base_clean = clean_domain_name(urlparse(base_url).netloc)
    sitemap_candidates = [
        urljoin(base_url, '/sitemap.xml'),
        urljoin(base_url, '/sitemap_index.xml')
    ]

    for sitemap_url in sitemap_candidates:
        try:
            if HAS_CURL_CFFI:
                resp = cffi_requests.get(sitemap_url, impersonate="chrome124", timeout=6)
            else:
                resp = requests.get(sitemap_url, headers=headers, timeout=6)
            if resp.status_code == 200 and '<?xml' in resp.text:
                root = ET.fromstring(resp.content)
                for elem in root.iter():
                    if elem.tag.endswith('loc') and elem.text:
                        loc = elem.text.strip()
                        # Handle localhost or misconfigured sitemap URLs by remapping to base_url
                        parsed_loc = urlparse(loc)
                        if 'localhost' in parsed_loc.netloc or '127.0.0.1' in parsed_loc.netloc:
                            loc = urljoin(base_url, parsed_loc.path)
                        if clean_domain_name(urlparse(loc).netloc) == base_clean:
                            clean_loc = loc.split('#')[0].rstrip('/')
                            if clean_loc not in discovered and clean_loc != base_url.rstrip('/'):
                                discovered.append(clean_loc)
        except Exception:
            pass

    return discovered

def scrape_url_content(url: str, crawl_depth: int = 5) -> tuple[str, str, str]:
    """
    Intelligent Deep Web Scraper:
    - Scrapes all content from Next.js, React, Nuxt, Vue, Angular, WordPress, and static sites.
    - Captures Header, Footer, Dropdown Menus, Sidebars, Addresses, Phone Numbers, Emails, Social Channels.
    - Crawls /sitemap.xml and probes standard subpages (/about, /contact-us, /privacy-policy, etc.).
    - Returns: (Title, Combined Structured Knowledge, Logo URL)
    """
    import time
    start_crawl = time.time()
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
        all_phones = set(main_meta.get('phones', []))
        all_emails = set(main_meta.get('emails', []))
        all_addresses = set(main_meta.get('addresses', []))

        # 1. Sitemap Discovery
        sitemap_urls = discover_sitemap_urls(url, headers)
        candidate_links = list(set(found_links + sitemap_urls))

        # 2. Standard Business Pages Probe (Guarantees contact, address, about, policy are discovered)
        standard_endpoints = [
            '/about', '/about-us', '/our-story', '/team',
            '/contact', '/contact-us', '/contact_us', '/t/contact_us', '/t/contact',
            '/about/contact', '/about/contact-us', '/help/contact', '/support/contact',
            '/reach-us', '/locations', '/office',
            '/services', '/solutions', '/products',
            '/pricing', '/plans', '/faq',
            '/privacy-policy', '/terms', '/terms-and-conditions'
        ]
        for ep in standard_endpoints:
            ep_url = urljoin(url, ep)
            if ep_url not in candidate_links and ep_url != url.rstrip('/'):
                candidate_links.append(ep_url)

        # 3. Prioritize Business & Contact Subpages
        priority_keywords = [
            'contact', 'address', 'location', 'office', 'about', 'team',
            'service', 'solution', 'pricing', 'policy', 'privacy', 'faq', 'feature'
        ]
        prioritized_links = []
        for kw in priority_keywords:
            for l in candidate_links:
                if kw in l.lower() and l not in prioritized_links and l != url.rstrip('/'):
                    prioritized_links.append(l)

        for l in candidate_links:
            if l not in prioritized_links and l != url.rstrip('/'):
                prioritized_links.append(l)

        # 4. Scrape Subpages up to crawl_depth with Multi-hop Link Discovery and 18s Max Elapsed Limit
        base_clean = clean_domain_name(domain)
        visited_urls = {url.rstrip('/')}
        crawl_queue = list(prioritized_links)
        while crawl_queue and len(collected_pages) < crawl_depth:
            if time.time() - start_crawl > 18:
                print(f"⏱️ [Web Crawler]: Max crawl duration (18s) reached. Finalizing with {len(collected_pages)} pages.")
                break
            sub_url = crawl_queue.pop(0)
            clean_sub_url = sub_url.split('#')[0].rstrip('/')
            if clean_sub_url in visited_urls:
                continue
            visited_urls.add(clean_sub_url)

            s_title, s_text, s_sub_links, s_meta = scrape_single_page(sub_url, headers)
            if s_text and len(s_text) > 120 and "404" not in s_title.lower() and "page not found" not in s_text.lower():
                collected_pages.append((sub_url, s_title, s_text))
                if s_meta.get('social_links'):
                    all_socials.update(s_meta['social_links'])
                if s_meta.get('phones'):
                    all_phones.update(s_meta['phones'])
                if s_meta.get('emails'):
                    all_emails.update(s_meta['emails'])
                if s_meta.get('addresses'):
                    all_addresses.update(s_meta['addresses'])

                # Multi-hop link discovery: check links discovered on subpages
                for link in s_sub_links:
                    clean_l = link.split('#')[0].rstrip('/')
                    if clean_domain_name(urlparse(clean_l).netloc) == base_clean and clean_l not in visited_urls:
                        # Prioritize contact, address, locations, about subpages
                        if any(kw in clean_l.lower() for kw in priority_keywords):
                            crawl_queue.insert(0, clean_l)
                        elif len(crawl_queue) < crawl_depth * 2:
                            crawl_queue.append(clean_l)

        print(f"🕸️ [Deep Web Crawler]: Scraped {len(collected_pages)} pages for domain {domain} with {len(all_addresses)} address locations and {len(all_phones)} phone channels.")

        # 5. Guaranteed Contact & Physical Address High-Priority Section
        overview_lines = [
            f"=== Organization & Website Profile: {main_title} ===",
            f"Official Website: {url}",
            f"Domain: {domain}"
        ]
        if main_meta.get('meta_desc'):
            overview_lines.append(f"Primary Mission & Summary: {main_meta['meta_desc']}")

        if all_addresses:
            overview_lines.append("\n=== Official Physical Address & Office Locations ===")
            for idx, addr in enumerate(all_addresses, 1):
                overview_lines.append(f"Office Location / Address {idx}: {addr}")
        else:
            overview_lines.append("\n=== Official Physical Address & Office Locations ===")
            overview_lines.append("No public physical street address listed on website. Please refer to official online support channels.")

        if all_phones or all_emails:
            overview_lines.append("\n=== Official Contact Channels ===")
            if all_phones:
                overview_lines.append(f"Telephone / Mobile Numbers: {', '.join(all_phones)}")
            if all_emails:
                overview_lines.append(f"Email Addresses: {', '.join(all_emails)}")
        else:
            overview_lines.append("\n=== Official Contact Channels ===")
            overview_lines.append("No direct public telephone or email listed on website. Please refer to official contact forms or support channels.")

        if all_socials:
            overview_lines.append("\n=== Verified Social Media & Follow Channels ===")
            for platform, s_url in all_socials.items():
                overview_lines.append(f"- {platform}: {s_url}")

        overview_block = "\n".join(overview_lines)

        # 6. Combine all pages into structured knowledge blocks
        formatted_sections = [overview_block]
        for p_url, p_title, p_content in collected_pages:
            formatted_sections.append(f"=== Web Page: {p_title} ({p_url}) ===\n{p_content}")

        combined_text = ("\n\n" + ("\n" + "="*50 + "\n\n").join(formatted_sections)).replace('\x00', '').replace('\0', '').strip()
        main_title = main_title.replace('\x00', '').replace('\0', '').strip()
        return main_title, combined_text, logo_url

    except Exception as e:
        raise Exception(f"Failed to scrape website: {str(e)}")
