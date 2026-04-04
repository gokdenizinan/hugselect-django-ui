
#!/usr/bin/env python3
"""
tech_noun_phrases.py

Pipeline:
  A) Preprocess/clean long technical text (remove code, logs, tables, stack traces)
  B) Extract noun phrases locally (spaCy noun chunks)
  C) Filter out non-technical noun phrases with heuristics
  D) (Optional) Send cleaned text to Gemini with rate-limit + caching

Install:
  pip install spacy regex
  python -m spacy download en_core_web_sm

Optional Gemini:
  pip install google-generativeai
  export GEMINI_API_KEY="..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set, Optional

import regex  # better unicode regex than re

# -----------------------------
# Preprocessing (noise removal)
# -----------------------------

CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]+`")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")

# Heuristics for "loggy" lines and "tabular" lines
STACKTRACE_HINT_RE = re.compile(r"^\s*(Traceback|File \"|at \w+\.|Exception:|ERROR|WARN|INFO)\b", re.IGNORECASE)
TABLEISH_LINE_RE = re.compile(r"^(\s*\|.*\|\s*|\s*[-+]{3,}\s*|\s*\d+(\s+\d+){3,}\s*)$")
MANY_NUMBERS_RE = re.compile(r"(\d[\d\.,%]*){6,}")  # lots of numeric tokens in one line
JSONISH_RE = re.compile(r"^\s*[\{\[]")  # line begins with { or [
YAMLY_RE = re.compile(r"^\s*\w[\w\-]*:\s+.+")  # key: value
SQLISH_RE = re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|WITH)\b", re.IGNORECASE)


def extract_code_blocks(text: str) -> List[str]:
    """Return list of triple-backtick fenced code blocks (content incl fences)."""
    return CODE_FENCE_RE.findall(text)


def strip_code_blocks(text: str) -> str:
    """Remove triple-backtick code blocks."""
    return CODE_FENCE_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_noisy_lines(text: str, keep_short_lines: bool = True) -> str:
    """
    Drop lines that look like stack traces, logs, heavy tables, or super-numeric output.
    Keeps prose lines.
    """
    out_lines = []
    for line in text.splitlines():
        raw = line.rstrip("\n")
        s = raw.strip()

        if not s:
            out_lines.append("")
            continue

        # drop obvious noise
        if STACKTRACE_HINT_RE.search(s):
            continue
        if TABLEISH_LINE_RE.match(s):
            continue
        if MANY_NUMBERS_RE.search(s):
            continue
        if SQLISH_RE.match(s):
            continue

        # drop JSON/YAML-like config dumps if very long
        if (JSONISH_RE.match(s) or YAMLY_RE.match(s)) and len(s) > 120:
            continue

        # optionally drop very short "bullet garbage"
        if not keep_short_lines and len(s) < 25 and not re.search(r"[A-Za-z]{6,}", s):
            continue

        out_lines.append(raw)

    return "\n".join(out_lines)


def preprocess_text(
    text: str,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_html: bool = True,
    remove_inline_code: bool = True,
    drop_noisy_lines: bool = True,
    keep_short_lines: bool = True,
) -> Tuple[str, Set[str]]:
    """
    Clean text and also extract "identifier-like" tokens from code blocks,
    which we can use to keep relevant noun phrases later.

    Returns: (cleaned_text, identifiers_from_code)
    """
    code_blocks = extract_code_blocks(text)
    identifiers = set()

    # Grab identifier-like tokens from code blocks (snake_case, camelCase, dotted.names, etc.)
    ident_re = regex.compile(r"\b[\p{L}_][\p{L}\p{N}_]*(?:\.[\p{L}_][\p{L}\p{N}_]*)*\b")
    for block in code_blocks:
        # strip fences for id extraction
        inner = re.sub(r"^```.*?\n|```$", "", block, flags=re.DOTALL).strip()
        for m in ident_re.finditer(inner):
            tok = m.group(0)
            # exclude common tiny tokens
            if len(tok) >= 3 and not tok.isdigit():
                identifiers.add(tok)

    # Remove code blocks entirely from prose input for noun phrase extraction
    cleaned = strip_code_blocks(text)

    if remove_inline_code:
        cleaned = INLINE_CODE_RE.sub(" ", cleaned)

    if remove_html:
        cleaned = HTML_TAG_RE.sub(" ", cleaned)

    if remove_urls:
        cleaned = URL_RE.sub(" ", cleaned)

    if remove_emails:
        cleaned = EMAIL_RE.sub(" ", cleaned)

    if drop_noisy_lines:
        cleaned = remove_noisy_lines(cleaned, keep_short_lines=keep_short_lines)

    cleaned = normalize_whitespace(cleaned)
    return cleaned, identifiers


# -----------------------------
# Noun phrase extraction (spaCy)
# -----------------------------

def load_spacy(model: str = "en_core_web_sm"):
    import spacy
    try:
        nlp = spacy.load(model, disable=["ner"])  # noun_chunks doesn't need NER
    except OSError as e:
        raise SystemExit(
            f"spaCy model '{model}' not found.\n"
            f"Install with:\n  python -m spacy download {model}\n"
        ) from e
    return nlp


def extract_noun_phrases_spacy(text: str, nlp, max_chars: int = 300_000) -> List[str]:
    """
    Extract noun chunks. For very large text, spaCy can slow down; we truncate by default.
    Better: chunk text by paragraphs if needed.
    """
    if len(text) > max_chars:
        text = text[:max_chars]

    doc = nlp(text)
    phrases = []
    for chunk in doc.noun_chunks:
        p = chunk.text.strip()
        # normalize quotes etc
        p = re.sub(r"\s+", " ", p)
        phrases.append(p)
    return phrases


# -----------------------------
# Filtering: keep technical terms
# -----------------------------

DEFAULT_DENY_PHRASES = {
    # expand as needed
    "this", "that", "these", "those",
    "the system", "the user", "this approach", "this method",
    "the result", "the results", "the analysis", "the data",
    "a lot", "a number", "the number", "some people", "some time",
}

TECH_HINT_KEYWORDS = {
    # broad technical vocabulary hints
    "api", "sdk", "cli", "http", "https", "tls", "ssl", "jwt", "oauth",
    "database", "schema", "index", "query", "sql", "nosql",
    "regression", "anova", "bayesian", "p-value", "confidence interval", "standard deviation",
    "gradient", "loss", "optimizer", "transformer", "embedding", "token", "vector",
    "docker", "kubernetes", "container", "pipeline", "ci/cd",
    "latency", "throughput", "cache", "memoization",
    "microservice", "endpoint", "serialization", "deserialization",
    "null pointer", "stack trace", "exception",
    "python", "java", "javascript", "typescript", "golang", "rust",
}

STOPWORD_RATIO_MAX = 0.6


def looks_like_identifier(s: str) -> bool:
    # snake_case / camelCase / dotted.name / has digits / has underscores
    if re.search(r"[A-Za-z]\w*_[A-Za-z0-9_]+", s):
        return True
    if re.search(r"\b[a-z]+[A-Z]\w*\b", s):  # camelCase
        return True
    if "." in s and re.search(r"\b[A-Za-z_]\w*(\.[A-Za-z_]\w*)+\b", s):
        return True
    if re.search(r"\b[A-Za-z_]\w*\d+\w*\b", s):
        return True
    return False


def technical_score(phrase: str, code_identifiers: Set[str]) -> int:
    """
    Simple heuristic scoring:
      +2 identifier-like
      +2 contains acronym or ALLCAPS token
      +2 contains digits/symbols common in tech terms (%, -, /)
      +2 overlaps code identifiers
      +1 contains tech hint keywords
    """
    p = phrase.strip()
    p_low = p.lower()

    score = 0

    if looks_like_identifier(p):
        score += 2

    # acronyms (e.g., "API", "HTTP", "ROC AUC")
    if re.search(r"\b[A-Z]{2,}\b", p):
        score += 2

    if re.search(r"[\d]|[%]|[/]|[-]", p):
        score += 2

    # overlap with identifiers extracted from code blocks
    # (either exact match, or phrase contains an identifier token)
    for tok in code_identifiers:
        if tok == p:
            score += 3
            break
        if tok in p:
            score += 2
            break

    # keyword hints
    for kw in TECH_HINT_KEYWORDS:
        if kw in p_low:
            score += 1
            break

    return score


def filter_noun_phrases(
    phrases: Iterable[str],
    code_identifiers: Set[str],
    min_len: int = 3,
    max_len: int = 80,
    min_score: int = 2,
    deny_phrases: Optional[Set[str]] = None,
) -> List[str]:
    """
    Remove generic phrases; keep technical ones.
    """
    deny_phrases = deny_phrases or set()
    deny = {d.lower() for d in (DEFAULT_DENY_PHRASES | deny_phrases)}

    # load stopwords from spaCy if available, else fallback
    try:
        import spacy
        stopwords = spacy.lang.en.stop_words.STOP_WORDS
    except Exception:
        stopwords = set(["the", "a", "an", "of", "to", "and", "in", "for", "on", "with", "by", "as"])

    seen = set()
    kept = []

    for p in phrases:
        raw = p.strip()
        if not raw:
            continue

        # normalize
        normalized = re.sub(r"\s+", " ", raw).strip(" \t\r\n\"'“”‘’.,;:()[]{}")
        if not normalized:
            continue

        low = normalized.lower()

        # length bounds
        if len(normalized) < min_len or len(normalized) > max_len:
            continue

        # reject if mostly stopwords
        toks = [t for t in re.split(r"\s+", low) if t]
        if toks:
            sw = sum(1 for t in toks if t in stopwords)
            if (sw / max(1, len(toks))) > STOPWORD_RATIO_MAX:
                continue

        # reject very generic phrases
        if low in deny:
            continue

        # reject single super-common words
        if len(toks) == 1 and toks[0] in stopwords:
            continue

        # compute score
        score = technical_score(normalized, code_identifiers)
        if score < min_score:
            continue

        if low not in seen:
            seen.add(low)
            kept.append(normalized)

    return kept


# -----------------------------
# Optional Gemini: rate-limit + caching + backoff
# -----------------------------

@dataclass
class GeminiConfig:
    model: str = "gemini-1.5-flash"  # change if needed
    rpm: int = 10                    # requests per minute cap you choose
    max_chars_per_chunk: int = 12_000
    cache_dir: Path = Path(".gemini_cache")


def chunk_text(text: str, max_chars: int) -> List[str]:
    """
    Chunk by paragraphs to stay under size limits.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0
    for p in paras:
        if current_len + len(p) + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [p]
            current_len = len(p)
        else:
            current.append(p)
            current_len += len(p) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def gemini_extract_noun_phrases(cleaned_text: str, cfg: GeminiConfig) -> List[str]:
    """
    Uses Gemini to extract noun phrases from cleaned text.
    Safe for free-tier-ish limits: chunking + caching + throttling + exponential backoff.

    Requires:
      pip install google-generativeai
      export GEMINI_API_KEY=...
    """
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY env var to use Gemini.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(cfg.model)

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    chunks = chunk_text(cleaned_text, cfg.max_chars_per_chunk)
    delay = 60.0 / max(1, cfg.rpm)
    last_call = 0.0

    all_phrases: List[str] = []
    for idx, chunk in enumerate(chunks):
        key = sha1(cfg.model + "::" + chunk)
        cache_path = cfg.cache_dir / f"{key}.json"

        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            all_phrases.extend(data.get("phrases", []))
            continue

        # throttle (simple RPM control)
        now = time.monotonic()
        wait = (last_call + delay) - now
        if wait > 0:
            time.sleep(wait)

        prompt = (
            "Extract a deduplicated list of noun phrases that are TECHNICAL (software, data science, "
            "statistics, engineering). Return JSON only, like:\n"
            "{ \"phrases\": [\"...\"] }\n\n"
            "Text:\n"
            f"{chunk}"
        )

        # exponential backoff for transient/rate limit errors
        backoff = 2.0
        for attempt in range(6):
            try:
                resp = model.generate_content(prompt)
                last_call = time.monotonic()

                # Gemini often returns text; we asked for JSON only, but we still guard parse.
                text = (resp.text or "").strip()
                # attempt to extract first JSON object
                m = re.search(r"\{.*\}", text, flags=re.DOTALL)
                if not m:
                    raise ValueError("No JSON object found in response.")
                obj = json.loads(m.group(0))
                phrases = obj.get("phrases", [])
                if not isinstance(phrases, list):
                    raise ValueError("JSON 'phrases' is not a list.")

                cache_path.write_text(json.dumps({"phrases": phrases}, ensure_ascii=False, indent=2), encoding="utf-8")
                all_phrases.extend([str(x) for x in phrases])
                break

            except Exception as e:
                # likely 429 / quota / transient
                if attempt == 5:
                    raise
                time.sleep(backoff)
                backoff *= 1.8

    # dedupe preserving order
    seen = set()
    out = []
    for p in all_phrases:
        low = p.strip().lower()
        if low and low not in seen:
            seen.add(low)
            out.append(p.strip())
    return out


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, required=True, help="Path to input .txt/.md")
    ap.add_argument("--spacy-model", type=str, default="en_core_web_sm")
    ap.add_argument("--min-score", type=int, default=2, help="Higher => stricter technical filtering")
    ap.add_argument("--use-gemini", action="store_true", help="Also call Gemini on cleaned text")
    ap.add_argument("--gemini-model", type=str, default="gemini-1.5-flash")
    ap.add_argument("--rpm", type=int, default=10, help="Requests per minute throttle for Gemini")
    args = ap.parse_args()

    raw = Path(args.infile).read_text(encoding="utf-8", errors="ignore")
    cleaned, code_ids = preprocess_text(raw)

    nlp = load_spacy(args.spacy_model)
    noun_phrases = extract_noun_phrases_spacy(cleaned, nlp)

    filtered = filter_noun_phrases(
        noun_phrases,
        code_identifiers=code_ids,
        min_score=args.min_score,
    )

    print("\n=== CLEANED TEXT (first 800 chars) ===\n")
    print(cleaned[:800] + ("..." if len(cleaned) > 800 else ""))

    print("\n=== LOCAL TECHNICAL NOUN PHRASES (filtered) ===\n")
    for p in filtered[:200]:
        print(p)
    if len(filtered) > 200:
        print(f"... ({len(filtered)} total)")

    if args.use_gemini:
        cfg = GeminiConfig(model=args.gemini_model, rpm=args.rpm)
        gemini_phrases = gemini_extract_noun_phrases(cleaned, cfg)

        # You can optionally re-filter Gemini output as well:
        gemini_filtered = filter_noun_phrases(gemini_phrases, code_ids, min_score=args.min_score)

        print("\n=== GEMINI TECHNICAL NOUN PHRASES (filtered) ===\n")
        for p in gemini_filtered[:200]:
            print(p)
        if len(gemini_filtered) > 200:
            print(f"... ({len(gemini_filtered)} total)")


if __name__ == "__main__":
    main()
