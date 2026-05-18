#!/usr/bin/env python3

from __future__ import annotations

import glob
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any

import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import sent_tokenize
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False
    stopwords = None
    WordNetLemmatizer = object

    def sent_tokenize(x: str) -> List[str]:
        return re.split(r"(?<=[.!?])\s+", x.strip()) if x else []

try:
    from transformers import AutoTokenizer
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False
    AutoTokenizer = None


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


PLACEHOLDERS = {"<URL>", "<CODE>", "<PATH>", "<EMAIL>"}


if _NLTK_AVAILABLE:
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    _BASE_STOPWORDS = set(stopwords.words("english"))
    STOP_WORDS = _BASE_STOPWORDS - {"not", "no", "nor", "never"}
    LEMMATIZER = WordNetLemmatizer()
else:
    STOP_WORDS = set()
    LEMMATIZER = None


@dataclass
class ReviewRecord:
    model_id: str
    source: str
    original: str
    processed: str


def load_mentioned_from_folder(folder_path: str) -> List[Dict[str, Any]]:
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


def generate_model_mentions(model_id: str) -> List[str]:
    if not model_id:
        return []

    variants = set()

    if "/" in model_id:
        org, model = model_id.split("/", 1)
    else:
        org, model = "", model_id

    variants |= {
        model_id,
        model,
        model.replace("-", " "),
        model.replace("_", " "),
    }

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

    return sorted({v.lower() for v in variants}, key=len, reverse=True)


def preprocess_text(text, model_id=None, normalize=False):
    if not isinstance(text, str):
        return ""

    text = text.strip().replace("\n", " ")

    if not text:
        return ""

    if model_id:
        for variant in generate_model_mentions(model_id):
            text = re.sub(
                rf"(?i)\b{re.escape(variant)}\b",
                "@@MODEL@@",
                text
            )

    text = re.sub(r"`[^`]+`|<code>.*?</code>", "<CODE>", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
    text = re.sub(r"\b\S+@\S+\.[A-Za-z]{2,}\b", "<EMAIL>", text)

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
        return s if s.startswith("<") and s.endswith(">") else "<PATH>"

    text = PATH_PATTERN.sub(_replace_path, text)

    text = re.sub(r"\s+", " ", text).strip()

    if normalize and _NLTK_AVAILABLE and LEMMATIZER is not None:
        words = re.findall(r"\b\w+\b|<URL>|<CODE>|<PATH>|<EMAIL>|@@MODEL@@", text)
        normalized_words = []

        for word in words:
            if word in {"<URL>", "<CODE>", "<PATH>", "<EMAIL>", "@@MODEL@@"}:
                normalized_words.append(word)
                continue

            w = word.lower()

            if w not in STOP_WORDS:
                normalized_words.append(LEMMATIZER.lemmatize(w))

        text = " ".join(normalized_words)

    if model_id:
        text = text.replace("@@MODEL@@", model_id)

    return text


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


def extract_name_snippet_tokens(
    text: str,
    name: str,
    max_tokens: int = 512,
    tokenizer_name: str = "roberta-base",
) -> str:
    if not _HF_AVAILABLE:
        return (text or "")[: max_tokens * 5]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    sentences = sent_tokenize(text)

    def _pack_from(start_idx: int) -> str:
        tokens: List[str] = []
        idx = start_idx

        while len(tokens) < max_tokens and idx < len(sentences):
            tokens.extend(tokenizer.tokenize(sentences[idx]))
            idx += 1

        return tokenizer.convert_tokens_to_string(tokens[:max_tokens])

    lower_name = name.lower()

    for i, sent in enumerate(sentences):
        if lower_name in sent.lower():
            toks = tokenizer.tokenize(sent)

            if len(toks) <= max_tokens:
                return _pack_from(i)

            name_toks = tokenizer.tokenize(name)
            pos = 0

            for j in range(len(toks) - len(name_toks) + 1):
                if toks[j:j + len(name_toks)] == name_toks:
                    pos = j
                    break

            start = max(0, pos - max_tokens // 2)
            end = min(len(toks), start + max_tokens)

            return tokenizer.convert_tokens_to_string(toks[start:end])

    toks = tokenizer.tokenize(text)
    return tokenizer.convert_tokens_to_string(toks[:max_tokens])


def collect_and_preprocess(
    mentioned: List[Dict[str, Any]],
    normalize: bool = False,
) -> List[ReviewRecord]:
    records: List[ReviewRecord] = []

    key_map = {
        "reddit": "reddit",
        "hf": "huggingface",
        "huggingface": "huggingface",
        "stack": "stack",
    }

    for model_dict in mentioned:
        model_id = model_dict.get("model_id", "")

        for input_key, output_source in key_map.items():
            for review in model_dict.get(input_key, []) or []:
                for raw in review.get("mentioned", []) or []:
                    processed = preprocess_text(
                        raw,
                        model_id=model_id,
                        normalize=normalize
                    )

                    if processed:
                        records.append(
                            ReviewRecord(
                                model_id=model_id,
                                source=output_source,
                                original=raw,
                                processed=processed,
                            )
                        )

    return records


def main():
    input_folder = "5-REVIEW_COLLECTION/united_f5"

    json_output_file = (
        "6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/"
        "preprocessing_sentiment_new.json"
    )

    csv_output_file = (
        "6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/"
        "preprocessing_sentiment_go_new.csv"
    )

    summary_output_file = (
        "6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/"
        "preprocessing_source_summary.csv"
    )

    normalize = False

    logger.info("Loading reviews from %s", input_folder)
    mentioned = load_mentioned_from_folder(input_folder)
    logger.info("Loaded %d model entries", len(mentioned))

    logger.info("Preprocessing reviews...")
    records = collect_and_preprocess(mentioned, normalize=normalize)
    logger.info("Produced %d records before filtering", len(records))

    filtered = filter_reviews(records)
    logger.info("%d records after filtering", len(filtered))

    payload = [r.__dict__ for r in filtered]

    os.makedirs(os.path.dirname(json_output_file), exist_ok=True)

    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Saved cleaned JSON to %s", json_output_file)

    df = pd.DataFrame(payload)

    duplicates_mask = df.duplicated(subset=["processed"], keep="first")

    df_unique = df[~duplicates_mask].reset_index(drop=True)
    df_duplicates = df[duplicates_mask].reset_index(drop=True)

    print("\nTotal rows before deduplication:", len(df))
    print("Unique rows after deduplication:", len(df_unique))
    print("Duplicate rows:", len(df_duplicates))

    print("\nRows per source BEFORE deduplication:")
    print(df["source"].value_counts())

    print("\nUnique models per source BEFORE deduplication:")
    print(df.groupby("source")["model_id"].nunique())

    print("\nRows per source AFTER deduplication:")
    print(df_unique["source"].value_counts())

    print("\nUnique models per source AFTER deduplication:")
    print(df_unique.groupby("source")["model_id"].nunique())

    summary = df_unique.groupby("source").agg(
        rows=("processed", "count"),
        unique_models=("model_id", "nunique"),
    ).reset_index()

    print("\nSummary after deduplication:")
    print(summary)

    df_unique["snippet"] = df_unique.apply(
        lambda row: extract_name_snippet_tokens(
            str(row["processed"]),
            str(row["model_id"])
        ),
        axis=1
    )

    # df_unique.to_csv(csv_output_file, index=False)
    # summary.to_csv(summary_output_file, index=False)

    logger.info("Saved unique CSV to %s", csv_output_file)
    logger.info("Saved source summary CSV to %s", summary_output_file)

    print("\ndone!")


if __name__ == "__main__":
    main()