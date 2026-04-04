#!/usr/bin/env python3
"""
Collect and triage peer-reviewed papers that (likely) use Hugging Face models.

Pipeline:
1) Query Semantic Scholar for candidate papers (2021+ by default)
2) Download open-access PDFs when available
3) Extract:
   - GitHub repo URLs
   - HF model IDs (org/model) or huggingface.co/org/model links
   - "rationale" signals (heuristic)
4) Validate HF model IDs exist on the Hub
5) Score + decide include/review/exclude
6) Save results to CSV/JSON

Docs:
- Semantic Scholar Academic Graph API: /graph/v1/paper/search :contentReference[oaicite:2]{index=2}
- HF model cards / Hub metadata: :contentReference[oaicite:3]{index=3}
"""

from __future__ import annotations
import argparse
import csv
import dataclasses
import json
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None  # type: ignore


S2_BASE = "https://api.semanticscholar.org/graph/v1"
DEFAULT_QUERY = (
    '("huggingface" OR "hugging face" OR "huggingface.co" OR "from_pretrained" '
    'OR "transformers" OR "diffusers") AND (code OR github OR repository)'
)

# --- Regex helpers ---

RE_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)

RE_GITHUB = re.compile(
    r"(https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", re.IGNORECASE
)

RE_HF_URL = re.compile(
    r"(https?://huggingface\.co/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+))",
    re.IGNORECASE,
)

# HF IDs sometimes appear as ORG/MODEL, but that pattern appears in other contexts too.
# We only accept ORG/MODEL if it's near HF context words.
RE_ORG_MODEL = re.compile(r"\b([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\b")
HF_CONTEXT = re.compile(r"(huggingface|hugging\s+face|transformers|diffusers|from_pretrained)", re.IGNORECASE)

# Double-underscore variant: CarperAI__FIM-NeoX-1.3B
RE_ORG__MODEL = re.compile(r"\b([A-Za-z0-9_.-]+)__([A-Za-z0-9_.-]+)\b")

# Rationale signal patterns (heuristic)
RATIONALE_PATTERNS = [
    re.compile(r"\bwe (choose|selected|select|use|adopt)\b.*\b(because|due to|to handle|to address)\b", re.IGNORECASE),
    re.compile(r"\b(pretrained|pre-trained)\b.*\b(checkpoint|model)\b", re.IGNORECASE),
    re.compile(r"\bwe (fine-?tune|finetune|fine tuned)\b", re.IGNORECASE),
    re.compile(r"\bimplementation details\b|\bexperimental setup\b|\btraining details\b", re.IGNORECASE),
]


@dataclasses.dataclass
class PaperRow:
    s2_paper_id: str
    title: str
    year: Optional[int]
    venue: str
    doi: str
    url: str
    pdf_url: str
    authors: str
    abstract: str

    github_repos: List[str] = dataclasses.field(default_factory=list)
    hf_models: List[str] = dataclasses.field(default_factory=list)
    rationale_hits: int = 0

    hf_models_valid: List[str] = dataclasses.field(default_factory=list)
    score: int = 0
    decision: str = "review"
    reason_codes: str = ""


def s2_get(path: str, params: Dict[str, Any], api_key: Optional[str]) -> Dict[str, Any]:
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    r = requests.get(f"{S2_BASE}{path}", params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_candidates_s2(
    query: str,
    year_from: int,
    limit: int,
    api_key: Optional[str],
    batch_size: int = 100,
    sleep_s: float = 0.4,
) -> List[PaperRow]:
    """
    Uses /graph/v1/paper/search
    """
    fields = ",".join([
        "title", "year", "venue", "authors", "abstract", "externalIds", "url",
        "openAccessPdf", "publicationTypes", "citationCount"
    ])

    out: List[PaperRow] = []
    offset = 0

    while len(out) < limit:
        take = min(batch_size, limit - len(out))
        data = s2_get(
            "/paper/search",
            params={
                "query": query,
                "limit": take,
                "offset": offset,
                "fields": fields,
            },
            api_key=api_key,
        )
        papers = data.get("data", [])
        if not papers:
            break

        for p in papers:
            year = p.get("year")
            if year is not None and year < year_from:
                continue

            ext = p.get("externalIds") or {}
            doi = ext.get("DOI") or ""
            # Some records miss DOI but PDF contains it; we’ll still keep for review.
            authors = ", ".join([a.get("name", "") for a in (p.get("authors") or []) if a.get("name")])

            open_pdf = (p.get("openAccessPdf") or {}).get("url") or ""
            out.append(PaperRow(
                s2_paper_id=p.get("paperId", ""),
                title=p.get("title", ""),
                year=year,
                venue=p.get("venue", "") or "",
                doi=doi,
                url=p.get("url", "") or "",
                pdf_url=open_pdf,
                authors=authors,
                abstract=p.get("abstract", "") or "",
            ))

        offset += len(papers)
        time.sleep(sleep_s)

    return out


def safe_get_text_from_pdf_url(pdf_url: str, max_bytes: int = 12_000_000) -> str:
    """
    Minimal PDF-to-text using pdftotext if available; fallback: try extracting text with pypdf.
    """
    if not pdf_url:
        return ""

    r = requests.get(pdf_url, timeout=45, stream=True)
    r.raise_for_status()

    content = b""
    for chunk in r.iter_content(chunk_size=64 * 1024):
        content += chunk
        if len(content) > max_bytes:
            break

    # Try pypdf first (pure python)
    try:
        from pypdf import PdfReader  # type: ignore
        import io
        reader = PdfReader(io.BytesIO(content))
        parts = []
        for i, page in enumerate(reader.pages[:10]):  # first 10 pages are enough for signals
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        pass

    return ""


def normalize_hf_id(org: str, model: str) -> str:
    org = org.strip()
    model = model.strip()
    return f"{org}/{model}"


def extract_signals(text: str) -> Tuple[List[str], List[str], int, str]:
    """
    Returns: (github_repos, hf_models, rationale_hits, doi_if_found)
    """
    github = sorted(set(m.group(1) for m in RE_GITHUB.finditer(text)))

    # HF URL matches are safest:
    hf = set()
    for m in RE_HF_URL.finditer(text):
        org = m.group(2)
        model = m.group(3)
        hf.add(normalize_hf_id(org, model))

    # Double underscore ids:
    for m in RE_ORG__MODEL.finditer(text):
        hf.add(normalize_hf_id(m.group(1), m.group(2)))

    # ORG/MODEL only when near HF context (to avoid ICML/2023 false hits)
    for m in RE_ORG_MODEL.finditer(text):
        candidate = normalize_hf_id(m.group(1), m.group(2))
        # Look around match for HF context
        start, end = m.span()
        window = text[max(0, start - 80): min(len(text), end + 80)]
        if HF_CONTEXT.search(window):
            hf.add(candidate)

    rationale_hits = sum(1 for pat in RATIONALE_PATTERNS if pat.search(text))

    doi_match = RE_DOI.search(text)
    doi_found = doi_match.group(0) if doi_match else ""

    return github, sorted(hf), rationale_hits, doi_found


def validate_hf_models(hf_ids: List[str], hf_token: Optional[str] = None) -> List[str]:
    if not hf_ids or HfApi is None:
        return []

    api = HfApi(token=hf_token) if hf_token else HfApi()
    valid = []
    for mid in hf_ids:
        try:
            info = api.model_info(mid)
            # If it exists, it’s valid
            if info and getattr(info, "modelId", None):
                valid.append(mid)
        except Exception:
            continue
    return valid


def score_and_decide(row: PaperRow) -> PaperRow:
    """
    Simple inclusion scoring adapted to HF-model usage:
    - Strong signals: repo + HF model IDs + rationale hits + DOI/venue/year
    """
    score = 0
    reasons = []

    if row.year and row.year >= 2021:
        score += 1
    if row.venue:
        score += 1
    if row.doi:
        score += 3
    else:
        reasons.append("NO_DOI")

    if row.github_repos:
        score += 4
    else:
        reasons.append("NO_REPO")

    if row.hf_models:
        score += 4
    else:
        reasons.append("NO_HF_MODEL_EVIDENCE")

    if row.hf_models_valid:
        score += 2

    score += min(row.rationale_hits, 3)  # cap

    # Decision thresholds (tune after you see initial results)
    if "NO_REPO" in reasons:
        decision = "exclude"
    elif score >= 12:
        decision = "include"
    elif score <= 7:
        decision = "exclude"
    else:
        decision = "review"

    row.score = score
    row.decision = decision
    row.reason_codes = ";".join(reasons)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--year-from", type=int, default=2021)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--s2-api-key", default=os.getenv("S2_API_KEY", ""))
    ap.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    ap.add_argument("--download-pdfs", action="store_true", help="Download and scan open-access PDFs when available")
    ap.add_argument("--out-csv", default="hf_case_study_candidates.csv")
    ap.add_argument("--out-json", default="hf_case_study_candidates.json")
    args = ap.parse_args()

    s2_key = args.s2_api_key or None
    hf_token = args.hf_token or None

    candidates = fetch_candidates_s2(
        query=args.query,
        year_from=args.year_from,
        limit=args.limit,
        api_key=s2_key,
    )

    results: List[PaperRow] = []
    for i, row in enumerate(candidates, 1):
        text = ""
        if args.download_pdfs and row.pdf_url:
            try:
                text = safe_get_text_from_pdf_url(row.pdf_url)
            except Exception:
                text = ""

        # If no PDF text, fall back to scanning abstract (lower recall)
        scan_text = text if text.strip() else (row.abstract or "")

        gh, hf, rh, doi_found = extract_signals(scan_text)
        row.github_repos = gh
        row.hf_models = hf
        row.rationale_hits = rh
        if not row.doi and doi_found:
            row.doi = doi_found

        # Validate HF IDs only if we found some
        row.hf_models_valid = validate_hf_models(row.hf_models, hf_token=hf_token)

        row = score_and_decide(row)
        results.append(row)

        if i % 25 == 0:
            print(f"[{i}/{len(candidates)}] processed…", file=sys.stderr)

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "s2_paper_id", "title", "year", "venue", "doi", "url", "pdf_url", "authors",
            "github_repos", "hf_models", "hf_models_valid", "rationale_hits",
            "score", "decision", "reason_codes"
        ])
        for r in results:
            w.writerow([
                r.s2_paper_id, r.title, r.year or "", r.venue, r.doi, r.url, r.pdf_url, r.authors,
                "|".join(r.github_repos),
                "|".join(r.hf_models),
                "|".join(r.hf_models_valid),
                r.rationale_hits,
                r.score, r.decision, r.reason_codes
            ])

    # Write JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.out_csv} and {args.out_json}", file=sys.stderr)

# pip install requests huggingface_hub pypdf
# export S2_API_KEY="..."   # optional but recommended
# export HF_TOKEN="..."     # optional (for higher HF rate limits)

# python collect_hf_case_studies.py --limit 400 --download-pdfs


if __name__ == "__main__":
    import os
    S2_API_KEY="8UOI7VYemR3XcVoytIIE21CPV4bAKjid977MCxjT"
    os.environ["S2_API_KEY"]= "8UOI7VYemR3XcVoytIIE21CPV4bAKjid977MCxjT"
    os.environ["HF_TOKEN"]= "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"
    HF_TOKEN= "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"

    main()
