#!/usr/bin/env python3
"""
Collect and triage peer-reviewed papers that (likely) use Hugging Face models.

Improvements over the original:
1) Supports multiple queries via CLI or direct Python calls
2) Deduplicates across query results
3) Deduplication prefers DOI over Semantic Scholar paper ID
4) Merges duplicate rows to preserve strongest signals
5) Keeps Semantic Scholar as the search backend
6) Saves both CSV and JSON
7) Makes exclusion of papers with no GitHub repo toggleable
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None  # type: ignore

try:
    from pypdf import PdfReader  # type: ignore
except ImportError:
    PdfReader = None  # type: ignore


S2_BASE = "https://api.semanticscholar.org/graph/v1"

DEFAULT_QUERY = (
    '("huggingface" OR "hugging face" OR transformers OR diffusers OR "pretrained model" OR checkpoint) '
    'AND (selected OR choose OR chose OR adopted OR "fine-tuned" OR evaluated) '
    'AND (classification OR detection OR segmentation OR "question answering" OR retrieval OR generation)'
)

RE_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
RE_GITHUB = re.compile(
    r"(https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
RE_HF_URL = re.compile(
    r"(https?://huggingface\.co/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+))",
    re.IGNORECASE,
)
RE_ORG__MODEL = re.compile(r"\b([A-Za-z0-9_.-]+)__([A-Za-z0-9_.-]+)\b")
RE_ORG_MODEL = re.compile(r"\b([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\b")

HF_CONTEXT = re.compile(
    r"(huggingface|hugging\s+face|transformers|diffusers|from_pretrained|hf_hub_download|snapshot_download)",
    re.IGNORECASE,
)

RATIONALE_PATTERNS = [
    re.compile(
        r"\bwe (choose|selected|select|use|adopt)\b.*\b(because|due to|to handle|to address)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(pretrained|pre-trained)\b.*\b(checkpoint|model)\b", re.IGNORECASE),
    re.compile(r"\bwe (fine-?tune|finetune|fine tuned)\b", re.IGNORECASE),
    re.compile(
        r"\bimplementation details\b|\bexperimental setup\b|\btraining details\b",
        re.IGNORECASE,
    ),
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
    citation_count: Optional[int] = None
    publication_types: List[str] = dataclasses.field(default_factory=list)
    fields_of_study: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class RunConfig:
    queries: List[str]
    year_from: int = 2021
    limit: int = 100
    s2_api_key: Optional[str] = None
    hf_token: Optional[str] = None
    download_pdfs: bool = False
    out_csv: str = "hf_case_study_candidates.csv"
    out_json: str = "hf_case_study_candidates.json"
    batch_size: int = 100
    sleep_s: float = 1.2
    exclude_no_repo: bool = True
    checkpoint_every: int = 25
    save_intermediate: bool = True
    verbose: bool = True


def s2_get(path: str, params: Dict[str, Any], api_key: Optional[str], timeout: int = 30) -> Dict[str, Any]:
    headers: Dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    response = requests.get(
        f"{S2_BASE}{path}",
        params=params,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def fetch_candidates_s2(
    query: str,
    year_from: int,
    limit: int,
    api_key: Optional[str],
    batch_size: int = 100,
    sleep_s: float = 5.2,
    
) -> List[PaperRow]:
    """
    Uses Semantic Scholar /graph/v1/paper/search with pagination.
    """
    fields = ",".join([
        "paperId",
        "title",
        "year",
        "venue",
        "authors",
        "abstract",
        "externalIds",
        "url",
        "openAccessPdf",
        "publicationTypes",
        "citationCount",
        "fieldsOfStudy",
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
            authors = ", ".join(
                a.get("name", "")
                for a in (p.get("authors") or [])
                if a.get("name")
            )
            open_pdf = (p.get("openAccessPdf") or {}).get("url") or ""
            citation_count = p.get("citationCount")
            publication_types = p.get("publicationTypes") or []
            fields_of_study = p.get("fieldsOfStudy") or []

            out.append(
                PaperRow(
                    s2_paper_id=p.get("paperId", "") or "",
                    title=p.get("title", "") or "",
                    year=year,
                    venue=p.get("venue", "") or "",
                    doi=doi,
                    url=p.get("url", "") or "",
                    pdf_url=open_pdf,
                    authors=authors,
                    abstract=p.get("abstract", "") or "",

                    citation_count=citation_count,
                    publication_types=publication_types,
                    fields_of_study=fields_of_study,
                )
            )

        offset += len(papers)
        time.sleep(sleep_s)

    return out


def safe_get_text_from_pdf_url(pdf_url: str, max_bytes: int = 12_000_000, max_pages: int = 10) -> str:
    """
    Downloads a PDF up to max_bytes and attempts text extraction.

    Returns empty string on failure.
    """
    if not pdf_url:
        return ""

    if PdfReader is None:
        return ""

    response = requests.get(pdf_url, timeout=45, stream=True)
    response.raise_for_status()

    content = bytearray()
    truncated = False

    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue

        remaining = max_bytes - len(content)
        if remaining <= 0:
            truncated = True
            break

        if len(chunk) > remaining:
            content.extend(chunk[:remaining])
            truncated = True
            break

        content.extend(chunk)

    if truncated:
        # Parsing truncated PDFs is unreliable; fail fast instead of silently
        # attempting to read partial binary content.
        return ""

    try:
        reader = PdfReader(io.BytesIO(bytes(content)))
        parts: List[str] = []
        for page in reader.pages[:max_pages]:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        return ""


def normalize_hf_id(org: str, model: str) -> str:
    return f"{org.strip()}/{model.strip()}"


def extract_signals(text: str) -> Tuple[List[str], List[str], int, str]:
    github = sorted(set(m.group(1) for m in RE_GITHUB.finditer(text)))

    hf = set()

    for m in RE_HF_URL.finditer(text):
        hf.add(normalize_hf_id(m.group(2), m.group(3)))

    for m in RE_ORG__MODEL.finditer(text):
        hf.add(normalize_hf_id(m.group(1), m.group(2)))

    for m in RE_ORG_MODEL.finditer(text):
        start, end = m.span()
        window = text[max(0, start - 80): min(len(text), end + 80)]
        if HF_CONTEXT.search(window):
            hf.add(normalize_hf_id(m.group(1), m.group(2)))

    rationale_hits = sum(1 for pat in RATIONALE_PATTERNS if pat.search(text))

    doi_match = RE_DOI.search(text)
    doi_found = doi_match.group(0) if doi_match else ""

    return github, sorted(hf), rationale_hits, doi_found


def validate_hf_models(hf_ids: List[str], hf_token: Optional[str] = None) -> List[str]:
    if not hf_ids or HfApi is None:
        return []

    api = HfApi(token=hf_token) if hf_token else HfApi()
    valid: List[str] = []

    for model_id in hf_ids:
        try:
            info = api.model_info(model_id)
            if info and getattr(info, "modelId", None):
                valid.append(model_id)
        except Exception:
            continue

    return valid


def score_and_decide(row: PaperRow, exclude_no_repo: bool = True) -> PaperRow:
    score = 0
    reasons: List[str] = []

    if row.year and row.year >= 2021:
        score += 1
    if row.citation_count and row.citation_count > 50:
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

    score += min(row.rationale_hits, 3)

    if exclude_no_repo and "NO_REPO" in reasons:
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


def norm_doi(doi: str) -> str:
    d = (doi or "").strip().lower()
    d = d.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return d


def norm_title(title: str) -> str:
    t = (title or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t


def dedupe_key(row: PaperRow) -> str:
    """
    Prefer DOI over Semantic Scholar ID for deduplication.

    This helps merge records that refer to the same paper even when they came
    back with different Semantic Scholar IDs across searches.
    """
    d = norm_doi(row.doi)
    if d:
        return f"doi:{d}"

    if row.s2_paper_id:
        return f"s2:{row.s2_paper_id}"

    first_author = row.authors.split(",")[0].strip().lower() if row.authors else ""
    return f"fp:{norm_title(row.title)}|{row.year or ''}|{first_author}"


def merge_rows(a: PaperRow, b: PaperRow, exclude_no_repo: bool = True) -> PaperRow:
    a.github_repos = sorted(set(a.github_repos) | set(b.github_repos))
    a.hf_models = sorted(set(a.hf_models) | set(b.hf_models))
    a.hf_models_valid = sorted(set(a.hf_models_valid) | set(b.hf_models_valid))

    if not a.doi and b.doi:
        a.doi = b.doi
    if not a.venue and b.venue:
        a.venue = b.venue
    if not a.url and b.url:
        a.url = b.url
    if not a.pdf_url and b.pdf_url:
        a.pdf_url = b.pdf_url
    if not a.abstract and b.abstract:
        a.abstract = b.abstract
    if not a.authors and b.authors:
        a.authors = b.authors
    if not a.year and b.year:
        a.year = b.year
    if not a.title and b.title:
        a.title = b.title
    if not a.s2_paper_id and b.s2_paper_id:
        a.s2_paper_id = b.s2_paper_id

    a.rationale_hits = max(a.rationale_hits, b.rationale_hits)
    return score_and_decide(a, exclude_no_repo=exclude_no_repo)

def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, file=sys.stderr, flush=True)
def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, file=sys.stderr, flush=True)
def process_queries(config: RunConfig) -> List[PaperRow]:
    all_by_key: Dict[str, PaperRow] = {}
    processed_total = 0

    log(
        f"Starting run: {len(config.queries)} queries | "
        f"limit/query={config.limit} | year_from={config.year_from}",
        config.verbose,
    )

    for qi, query in enumerate(config.queries, start=1):
        time.sleep(1)
        log(f"\n=== Query {qi}/{len(config.queries)} ===", config.verbose)
        log(f"Query string: {query}", config.verbose)

        candidates = fetch_candidates_s2(
            query=query,
            year_from=config.year_from,
            limit=config.limit,
            api_key=config.s2_api_key,
            batch_size=config.batch_size,
            sleep_s=config.sleep_s,
        )

        log(
            f"Fetched {len(candidates)} candidates for query {qi}",
            config.verbose,
        )

        for i, row in enumerate(candidates, start=1):
            text = ""

            if config.download_pdfs and row.pdf_url:
                try:
                    text = safe_get_text_from_pdf_url(row.pdf_url)
                except Exception as e:
                    log(
                        f"[query {qi} item {i}] PDF download/parse failed: {e}",
                        config.verbose,
                    )
                    text = ""

            scan_text = text if text.strip() else (row.abstract or "")

            github_repos, hf_models, rationale_hits, doi_found = extract_signals(scan_text)
            row.github_repos = github_repos
            row.hf_models = hf_models
            row.rationale_hits = rationale_hits

            if not row.doi and doi_found:
                row.doi = doi_found

            row.hf_models_valid = validate_hf_models(row.hf_models, hf_token=config.hf_token)
            row = score_and_decide(row, exclude_no_repo=config.exclude_no_repo)

            key = dedupe_key(row)
            if key in all_by_key:
                all_by_key[key] = merge_rows(
                    all_by_key[key],
                    row,
                    exclude_no_repo=config.exclude_no_repo,
                )
            else:
                all_by_key[key] = row

            processed_total += 1

            log(
                f"[query {qi}/{len(config.queries)} | item {i}/{len(candidates)} | total {processed_total}] "
                f"title={row.title[:80]!r} | score={row.score} | decision={row.decision} | "
                f"repos={len(row.github_repos)} | hf={len(row.hf_models)} | unique={len(all_by_key)}",
                config.verbose and (i <= 5 or i % 10 == 0),
            )

            if config.save_intermediate and (processed_total % config.checkpoint_every == 0):
                log(
                    f"Checkpoint save at total={processed_total} "
                    f"(unique papers={len(all_by_key)})",
                    config.verbose,
                )
                write_outputs(list(all_by_key.values()), config.out_csv, config.out_json)

        if config.save_intermediate:
            log(
                f"End-of-query save for query {qi} "
                f"(unique papers={len(all_by_key)})",
                config.verbose,
            )
            write_outputs(list(all_by_key.values()), config.out_csv, config.out_json)

    log(
        f"\nFinished. Processed total={processed_total}, unique={len(all_by_key)}",
        config.verbose,
    )
    return list(all_by_key.values())


def write_outputs(results: List[PaperRow], out_csv: str, out_json: str) -> None:
    csv_dir = os.path.dirname(out_csv)
    json_dir = os.path.dirname(out_json)

    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "s2_paper_id",
                "title",
                "year",
                "venue",
                "doi",
                "url",
                "pdf_url",
                "authors",
                "abstract",
                "github_repos",
                "hf_models",
                "hf_models_valid",
                "rationale_hits",
                "score",
                "decision",
                "reason_codes",
                "citation_count",
                "publication_types",
                "fields_of_study",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.s2_paper_id,
                    row.title,
                    row.year or "",
                    row.venue,
                    row.doi,
                    row.url,
                    row.pdf_url,
                    row.authors,
                    row.abstract,
                    "|".join(row.github_repos),
                    "|".join(row.hf_models),
                    "|".join(row.hf_models_valid),
                    row.rationale_hits,
                    row.score,
                    row.decision,
                    row.reason_codes,
                    row.citation_count or "",
                    "|".join(row.publication_types),
                    "|".join(row.fields_of_study),
                ]
            )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(r) for r in results], f, ensure_ascii=False, indent=2)


def collect_papers(
    queries: Optional[List[str]] = None,
    year_from: int = 2021,
    limit: int = 100,
    s2_api_key: Optional[str] = None,
    hf_token: Optional[str] = None,
    download_pdfs: bool = False,
    out_csv: str = "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/hf_case_study_candidates.csv",
    out_json: str = "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/hf_case_study_candidates.json",
    batch_size: int = 100,
    sleep_s: float = 5.2,
    exclude_no_repo: bool = True,
    checkpoint_every: int = 25,
    save_intermediate: bool = True,
    verbose: bool = True,
) -> List[PaperRow]:
    """
    Reusable Python API.

    Example:
        results = collect_papers(
            queries=[
                "huggingface AND classification",
                "transformers AND fine-tuned"
            ],
            limit=200,
            exclude_no_repo=False,
        )
    """
    config = RunConfig(
        queries=queries or [DEFAULT_QUERY],
        year_from=year_from,
        limit=limit,
        s2_api_key=(s2_api_key or os.getenv("S2_API_KEY", "")).strip() or None,
        hf_token=(hf_token or os.getenv("HF_TOKEN", "")).strip() or None,
        download_pdfs=download_pdfs,
        out_csv=out_csv,
        out_json=out_json,
        batch_size=batch_size,
        sleep_s=sleep_s,
        exclude_no_repo=exclude_no_repo,
    )

    results = process_queries(config)
    write_outputs(results, config.out_csv, config.out_json)
    return results


from A_queries import  queries_100_broad, queries_100_specific, queries_domain, queries_domain_2
if __name__ == "__main__":
    results = collect_papers(
        queries=queries_domain_2,
        limit=100,
        s2_api_key="PKmmBIFgDr6FkomlPibjQ862kKxfGHyk1CKpKRPs",
        hf_token="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",
        exclude_no_repo=False,
        out_csv = "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_5.csv",
        out_json= "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_5.json",
    )

    print(f"Collected {len(results)} papers")