import json
import time
import random
import requests
from tqdm import tqdm
from huggingface_hub import HfApi
from bs4 import BeautifulSoup  # pip install beautifulsoup4
from huggingface_hub import hf_hub_download
import re
from huggingface_hub.utils import get_session
from huggingface_hub.utils import get_session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os

api = HfApi(token="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")  

### HTLM EXTRACTION
def clean_and_filter_chunks(md: str) -> str:
    """
    Convert Markdown to plain text, then join only the chunks
    that contain >=3 words (even if they have URLs).
    """
    # turn to text strings
    chunks = list(BeautifulSoup(md, "html.parser").stripped_strings)

    cleaned = []
    for c in chunks:
        # strip markdown syntax and stray quotes
        t = re.sub(r'^[#>*\-\+\s]+', '', c)
        t = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', t)
        t = re.sub(r'`([^`]+)`', r'\1', t)
        t = re.sub(r'[\r\n\t]+', ' ', t)
        t = t.replace('\\', '').replace('"', '').replace("'", '')
        t = re.sub(r'\s{2,}', ' ', t).strip()
        if len(t.split()) >= 3:
            cleaned.append(t)

    return " ".join(cleaned)

def extract_description_from_readme(text: str, model_name: str) -> str:
    """
    1) Strip YAML front-matter
    2) Force blank line before headings
    3) Drop code blocks / tables / HTML
    4) Remove leading headings
    5) Split into paragraphs
    6) Drop title paragraph, then drop any "gating" paras
    7) Return the remaining paragraphs joined by blank lines
    """
    # 1) Remove YAML front-matter
    text = re.sub(r'(?ms)^---\s*\n.*?\n---\s*\n', '', text)

    # 2) Ensure headings start a new paragraph
    text = re.sub(r'(?m)^(#{1,6}\s)', r'\n\n\1', text)

    # 3) Drop fenced code, tables, HTML
    text = re.sub(r'(?ms)^```.*?```', '', text)
    text = re.sub(r'(?m)^\|.*\|$', '', text)
    text = re.sub(r'(?ms)<[^>]+>', '', text)

    # 4) Remove any leading Markdown headings
    lines = text.splitlines()
    while lines and re.match(r'^#{1,6}\s*', lines[0]):
        lines.pop(0)
    text = "\n".join(lines)

    # 5) Split into paragraphs
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    # 6a) Drop a first paragraph that’s literally just the model name/title
    if paras and paras[0].lower().endswith(model_name.lower()):
        paras = paras[1:]

    # 6b) Drop any “gating” paragraphs (login/sign-up/access notices)
    gating_patterns = [
        r'\bagree\b',
        r'\blogin\b',
        r'\bsign[\s-]?up\b',
        r'\baccess this model\b',
        r'\bconditions?\b',
        r'\bUse this model\b',
        r'\bYou need to\b'
    ]
    filtered = [
        p for p in paras
        if not any(re.search(pat, p, re.I) for pat in gating_patterns)
    ]
    # if we filtered *everything*, fall back to original list
    paras = filtered or paras

    return "\n\n".join(paras)

def fetch_and_clean_description(session, model_id: str) -> str:
    desc = ""

    # 1) Try README.md
    try:
        # resp = session.get(f"https://huggingface.co/{model_id}/raw/main/README.md")
        resp = session.get(f"https://huggingface.co/{model_id}/resolve/main/README.md")
        resp.raise_for_status()
        raw_md = extract_description_from_readme(resp.text, model_id.split("/")[-1])
        if raw_md:
            desc = clean_and_filter_chunks(raw_md)
    except Exception:
        desc = ""

    # 2) Fallback to model-card API
    if not desc:
        card_url = f"https://huggingface.co/api/models/{model_id}/card"
        card_resp = session.get(card_url)
        if card_resp.ok:
            content = card_resp.json().get("data", {}).get("content", "")
            raw_md = extract_description_from_readme(content, model_id.split("/")[-1])
            if raw_md:
                desc = clean_and_filter_chunks(raw_md)

    return desc

# --- Main logic ---
with open("1-MODEL_FILTERING/no_desc.json", "r", encoding="utf-8") as f:
    model_ids = json.load(f)


HF_TOKEN = os.getenv("HF_TOKEN", "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")  # fallback to hard-coded token

# 2) Create session
session = requests.Session()
session.headers.update({"Authorization": f"Bearer {HF_TOKEN}"})

# 3) Setup retries
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD","GET","OPTIONS"]  # for requests >= v2.26
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)

print("starto...")

# Fetch descriptions
for model_id in tqdm(model_ids, desc="Collecting model descriptions"):
    desc = fetch_and_clean_description(session, model_id)
    descriptions = {
        "model_id": model_id,
        "description": desc
    }
    model_id_sav = model_id.replace("/", "__")
    with open(f"1-MODEL_FILTERING/missing_descriptions/{model_id_sav}.json", "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)

print(f"✅ Saved models with descriptions to model_descriptions.json")




# print(f"✅ Collected stats for {len(descriptions)}/{len(model_ids)} models, missing {len(gone_models)}")
