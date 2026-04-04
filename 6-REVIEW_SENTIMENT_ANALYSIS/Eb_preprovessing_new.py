#!/usr/bin/env python3
"""
AI Review Preprocessing — Refactored Script

What this script does
---------------------
1) Loads JSON files with model review data from a folder.
2) Preprocesses each review, preserving model IDs while anonymizing URLs/emails/paths/code.
3) (Optionally) normalizes text for classical ML pipelines (lowercase + lemmatize + stopword removal).
4) Filters out reviews that are only or mostly placeholders.
5) Saves a clean JSON for downstream sentiment analysis.

Usage
-----
python preprocess_reviews.py \
  --input-folder 5-REVIEW_COLLECTION/united_f5 \
  --output-file 6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/preprocessing_sentiment_new.json \
  --normalize  # optional

Notes
-----
- Keeps negations (not/no/nor/never) in stopwords.
- Uses a unique sentinel @@MODEL@@ to protect model identifiers during cleaning.
- Includes a utility to extract a token-limited snippet around a name using a HuggingFace tokenizer (optional).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any
import pandas as pd

# --- Optional heavy deps (nltk + transformers) ---------------------------------
# These sections are optional. The script will still run without --normalize or
# without the snippet extraction utility if the packages/corpora are missing.

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize, sent_tokenize
    _NLTK_AVAILABLE = True
except Exception:  # pragma: no cover
    _NLTK_AVAILABLE = False
    stopwords = None
    WordNetLemmatizer = object  # type: ignore
    def word_tokenize(x: str) -> List[str]:  # type: ignore
        return x.split()
    def sent_tokenize(x: str) -> List[str]:  # type: ignore
        return re.split(r"(?<=[.!?])\s+", x.strip()) if x else []

try:
    from transformers import AutoTokenizer
    _HF_AVAILABLE = True
except Exception:  # pragma: no cover
    _HF_AVAILABLE = False
    AutoTokenizer = None  # type: ignore

# --- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration --------------------------------------------------------------
# Placeholders to detect/remove-only texts
PLACEHOLDERS = {"<URL>", "<CODE>", "<PATH>", "<EMAIL>"}

# Keep negations if NLTK is available
if _NLTK_AVAILABLE:
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

    _BASE_STOPWORDS = set(stopwords.words("english"))
    STOP_WORDS = _BASE_STOPWORDS - {"not", "no", "nor", "never"}
    LEMMATIZER = WordNetLemmatizer()
else:
    STOP_WORDS = set()
    LEMMATIZER = None  # type: ignore

# --- Data structures ------------------------------------------------------------
@dataclass
class ReviewRecord:
    model_id: str
    original: str
    processed: str

# --- I/O -----------------------------------------------------------------------

def load_mentioned_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """Load all *.json files; each file may contain a dict or a list of dicts."""
    mentioned: List[Dict[str, Any]] = []
    for file_path in glob.glob(os.path.join(folder_path, "*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    mentioned.extend(data)
                else:
                    mentioned.append(data)
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON: %s", file_path)
        except OSError as e:
            logger.warning("Could not open %s: %s", file_path, e)
    return mentioned

# --- Model name handling --------------------------------------------------------

def generate_model_mentions(model_id: str) -> List[str]:
    """Generate common variants of a model identifier, lowercased and sorted by length.

    Examples: "org/model-name" -> ["org/model-name", "model-name", "model name", "org model-name", ...]
    """
    if not model_id:
        return []

    variants = set()
    if "/" in model_id:
        org, model = model_id.split("/", 1)
    else:
        org, model = "", model_id

    variants |= {model_id, model, model.replace("-", " "), model.replace("_", " ")}
    if org:
        variants |= {
            org,
            f"{org}/{model}",
            f"{org} {model}",
            f"{org}-{model}",
            f"{org}:{model}",
            f"{org}.{model}",
            f"{org} {model.replace('-', ' ')}",
        }
    # Normalize to lowercase for matching
    return sorted({v.lower() for v in variants}, key=len, reverse=True)

# --- Core preprocessing ---------------------------------------------------------

def preprocess_text(text, model_id=None, normalize=False):
    if not isinstance(text, str):
        return ""

    text = text.strip().replace("\n", " ")
    if not text:
        return ""

    # Protect model IDs
    if model_id:
        for variant in generate_model_mentions(model_id):
            text = re.sub(rf"(?i)\b{re.escape(variant)}\b", "@@MODEL@@", text)

    # --- Safe replacements: code/URL/email first ---
    text = re.sub(r"`[^`]+`|<code>.*?</code>", "<CODE>", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
    text = re.sub(r"\b\S+@\S+\.[A-Za-z]{2,}\b", "<EMAIL>", text)

    # --- Safer path replacement ---
    WINDOWS_PATH = r"(?:\b[A-Za-z]:\\)(?:[^\\\s]+\\)+[^\\\s]*"
    UNIX_ABS_PATH = r"(?:(?<=\s)|^)/(?:[^/\s]+/)+[^/\s]*"
    HOME_PATH = r"(?:(?<=\s)|^)~/(?:[^/\s]+/)+[^/\s]*"
    REL_PATH = r"(?:(?<=\s)|^)(?:\./|\.\./)(?:[^/\s]+/)+[^/\s]*"
    PATH_PATTERN = re.compile(
        rf"(?:{WINDOWS_PATH})|(?:{UNIX_ABS_PATH})|(?:{HOME_PATH})|(?:{REL_PATH})",
        flags=re.VERBOSE,
    )

    def _replace_path(m):
        s = m.group(0)
        return s if (s.startswith("<") and s.endswith(">")) else "<PATH>"

    text = PATH_PATTERN.sub(_replace_path, text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # (Optional) normalization...
    # ...

    # Restore model id
    if model_id:
        text = text.replace("@@MODEL@@", model_id)

    return text


# --- Filtering -----------------------------------------------------------------

def mostly_placeholders(text: str, threshold: float = 0.9) -> bool:
    if not text:
        return True
    words = text.split()
    count_placeholders = sum(1 for w in words if w in PLACEHOLDERS)
    return (count_placeholders / max(1, len(words))) >= threshold


def filter_reviews(reviews: Iterable[ReviewRecord]) -> List[ReviewRecord]:
    out: List[ReviewRecord] = []
    for r in reviews:
        proc = r.processed or ""
        if proc in PLACEHOLDERS:
            continue
        if mostly_placeholders(proc):
            continue
        out.append(r)
    return out

# --- Snippet extraction (optional) ---------------------------------------------

def extract_name_snippet_tokens(
    text: str,
    name: str,
    max_tokens: int = 512,
    tokenizer_name: str = "roberta-base",
) -> str:
    """Return a ~max_tokens snippet centered on `name` (if found). Requires transformers.
    If transformers are not available, returns the leading characters as a fallback.
    """
    if not _HF_AVAILABLE:
        return (text or "")[: max_tokens * 5]  # rough char fallback

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    sentences = sent_tokenize(text)

    def _pack_from(start_idx: int) -> str:
        tokens: List[str] = []
        idx = start_idx
        while len(tokens) < max_tokens and idx < len(sentences):
            tokens.extend(tokenizer.tokenize(sentences[idx]))
            idx += 1
        return tokenizer.convert_tokens_to_string(tokens[:max_tokens])

    # Try to find the sentence with the name
    lower_name = name.lower()
    for i, sent in enumerate(sentences):
        if lower_name in sent.lower():
            toks = tokenizer.tokenize(sent)
            if len(toks) <= max_tokens:
                return _pack_from(i)
            # too long; center around first match of name tokens
            name_toks = tokenizer.tokenize(name)
            pos = 0
            for j in range(len(toks) - len(name_toks) + 1):
                if toks[j : j + len(name_toks)] == name_toks:
                    pos = j
                    break
            start = max(0, pos - max_tokens // 2)
            end = min(len(toks), start + max_tokens)
            return tokenizer.convert_tokens_to_string(toks[start:end])

    # fallback: from start
    toks = tokenizer.tokenize(text)
    return tokenizer.convert_tokens_to_string(toks[:max_tokens])

# --- Pipeline runner ------------------------------------------------------------

def collect_and_preprocess(
    mentioned: List[Dict[str, Any]],
    normalize: bool = False,
) -> List[ReviewRecord]:
    records: List[ReviewRecord] = []
    keys = ("reddit", "hf", "stack")
    for model_dict in mentioned:
        model_id = model_dict.get("model_id", "")
        for key in keys:
            for review in model_dict.get(key, []) or []:
                for raw in review.get("mentioned", []) or []:
                    processed = preprocess_text(raw, model_id=model_id, normalize=normalize)
                    if processed:
                        records.append(ReviewRecord(model_id=model_id, original=raw, processed=processed))
    return records

# --- CLI -----------------------------------------------------------------------

def main():
    # --- User-configurable settings ---
    input_folder = "5-REVIEW_COLLECTION/united_f5"
    output_file = "6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/preprocessing_sentiment_new.json"
    normalize = False  # Set to True to apply lemmatization + stopword removal
    
    logger.info("Loading reviews from %s", input_folder)
    mentioned = load_mentioned_from_folder(input_folder)
    logger.info("Loaded %d model entries", len(mentioned))

    logger.info("Preprocessing reviews (normalize=%s)...", normalize)
    records = collect_and_preprocess(mentioned, normalize=normalize)
    logger.info("Produced %d records before filtering", len(records))

    filtered = filter_reviews(records)
    logger.info("%d records after filtering", len(filtered))

    # Save results
    payload = [r.__dict__ for r in filtered]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Saved cleaned data to %s", output_file)



if __name__ == "__main__":
    main()

PHASE = 12

if PHASE ==11:
    # bu promptlu hali llm le grade labeli cikartmak icin # ustteki dictionary alttaki list
    # with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/check_meaning_output_V0.json", "r", encoding = "utf-8") as f:
    #     prompt_f = json.load(f)
    
    with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/preprocessing_sentiment_new.json", "r", encoding = "utf-8") as f:
        prompt_f = json.load(f)

    rows = []
    # for _, entry in prompt_f.items():
    for entry in prompt_f:
        model_id = entry.get("model_id")
        processed = entry.get("processed")
        
        # Extract the integer grade from the "output" field
        output_text = entry.get("output", "")
        match = re.search(r'"grade"\s*:\s*(\d+)', output_text)
        grade = int(match.group(1)) if match else None

        rows.append({
            "model_id": model_id,
            "processed": processed,
            # "grade": grade #*10
        })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Identify duplicates based on the 'mentioned' column
    duplicates_mask = df.duplicated(subset=["processed"], keep="first")

    # DataFrame with unique reviews
    df_unique = df[~duplicates_mask].reset_index(drop=True)

    # DataFrame with duplicate reviews
    df_duplicates = df[duplicates_mask].reset_index(drop=True)

    print("Total:", len(df))
    print("Unique:", len(df_unique))
    print("Duplicate:", len(df_duplicates))

    df_unique['snippet'] = df_unique.apply(
    lambda row: extract_name_snippet_tokens(str(row['processed']), row['model_id']),
    axis=1
)

    # Filter rows where the two columns are different
    diff_rows = df_unique[df_unique["snippet"] != df_unique["processed"]]

    print(diff_rows)

    print(df)
    df_unique.to_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/preprocessing_sentiment_go_new.csv", index=False)

print("done!")
