#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
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


OA_BASE = "https://api.openalex.org"


DEFAULT_QUERY = (
    '("huggingface" OR "hugging face" OR transformers OR diffusers OR "pretrained model" OR checkpoint) '
    'AND (selected OR choose OR chose OR adopted OR "fine-tuned" OR evaluated) '
    'AND (classification OR detection OR segmentation OR "question answering" OR retrieval OR generation)'
)


# --- Regex helpers ---
RE_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
RE_GITHUB = re.compile(r"(https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", re.IGNORECASE)
RE_HF_URL = re.compile(r"(https?://huggingface\.co/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+))", re.IGNORECASE)
RE_ORG__MODEL = re.compile(r"\b([A-Za-z0-9_.-]+)__([A-Za-z0-9_.-]+)\b")
RE_ORG_MODEL = re.compile(r"\b([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\b")
HF_CONTEXT = re.compile(r"(huggingface|hugging\s+face|transformers|diffusers|from_pretrained|hf_hub_download|snapshot_download)",
                        re.IGNORECASE)

RATIONALE_PATTERNS = [
    re.compile(r"\bwe (choose|selected|select|use|adopt)\b.*\b(because|due to|to handle|to address)\b", re.IGNORECASE),
    re.compile(r"\b(pretrained|pre-trained)\b.*\b(checkpoint|model)\b", re.IGNORECASE),
    re.compile(r"\bwe (fine-?tune|finetune|fine tuned)\b", re.IGNORECASE),
    re.compile(r"\bimplementation details\b|\bexperimental setup\b|\btraining details\b", re.IGNORECASE),
]


@dataclasses.dataclass
class PaperRow:
    openalex_id: str
    title: str
    year: Optional[int]
    venue: str
    doi: str
    landing_page_url: str
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


def openalex_get(path: str, params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    r = requests.get(f"{OA_BASE}{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def decode_abstract(abstract_inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    OpenAlex often stores abstracts as an 'abstract_inverted_index' (token -> positions).
    Reconstruct it into plain text.
    """
    if not abstract_inverted_index:
        return ""
    pos_to_token: Dict[int, str] = {}
    for token, positions in abstract_inverted_index.items():
        for p in positions:
            pos_to_token[p] = token
    if not pos_to_token:
        return ""
    return " ".join(pos_to_token[i] for i in range(min(pos_to_token), max(pos_to_token) + 1) if i in pos_to_token)


def fetch_candidates_openalex(
    search_query: str,
    year_from: int,
    limit: int,
    mailto: Optional[str] = None,
    per_page: int = 200,
    sleep_s: float = 0.25,
) -> List[PaperRow]:
    """
    Uses OpenAlex /works endpoint with search + filter + paging.
    Docs: search, filter, paging. :contentReference[oaicite:4]{index=4}
    """
    # OpenAlex doesn't support clean numeric ranges for publication_year everywhere, so OR years.
    current_year = time.gmtime().tm_year
    years = [str(y) for y in range(year_from, current_year + 1)]
    year_filter = "|".join(years)

    out: List[PaperRow] = []
    page = 1

    # Minimal-ish fields are returned by default; abstract comes as inverted index when available. :contentReference[oaicite:5]{index=5}
    while len(out) < limit:
        take = min(per_page, limit - len(out))
        params: Dict[str, Any] = {
            "search": search_query,
            "filter": f"publication_year:{year_filter}",
            "per-page": take,
            "page": page,
        }
        if mailto:
            params["mailto"] = mailto  # helps with "polite pool" / reliability
        data = openalex_get("/works", params=params)

        results = data.get("results", [])
        if not results:
            break

        for w in results:
            # Prefer best OA pdf if present
            pdf_url = ""
            best_oa = w.get("best_oa_location") or {}
            if isinstance(best_oa, dict):
                pdf_url = best_oa.get("pdf_url") or ""
            if not pdf_url:
                primary_loc = w.get("primary_location") or {}
                if isinstance(primary_loc, dict):
                    pdf_url = primary_loc.get("pdf_url") or ""

            venue = ""
            hv = w.get("host_venue") or {}
            if isinstance(hv, dict):
                venue = hv.get("display_name") or ""

            authorships = w.get("authorships") or []
            author_names = []
            for a in authorships[:12]:
                aa = a.get("author") or {}
                name = aa.get("display_name")
                if name:
                    author_names.append(name)
            authors = ", ".join(author_names)

            doi = (w.get("doi") or "").replace("https://doi.org/", "").strip()
            abstract = decode_abstract(w.get("abstract_inverted_index"))

            out.append(PaperRow(
                openalex_id=w.get("id", ""),
                title=w.get("title", "") or "",
                year=w.get("publication_year"),
                venue=venue or "",
                doi=doi,
                landing_page_url=(w.get("primary_location") or {}).get("landing_page_url", "") or "",
                pdf_url=pdf_url,
                authors=authors,
                abstract=abstract,
            ))

        page += 1
        time.sleep(sleep_s)  # stay comfortably under rate limits :contentReference[oaicite:6]{index=6}

    return out


def safe_get_text_from_pdf_url(pdf_url: str, max_bytes: int = 12_000_000) -> str:
    if not pdf_url:
        return ""
    r = requests.get(pdf_url, timeout=45, stream=True)
    r.raise_for_status()

    content = b""
    for chunk in r.iter_content(chunk_size=64 * 1024):
        content += chunk
        if len(content) > max_bytes:
            break

    try:
        from pypdf import PdfReader  # type: ignore
        import io
        reader = PdfReader(io.BytesIO(content))
        parts = []
        for page in reader.pages[:10]:
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

    # ORG/MODEL only when surrounded by HF context to avoid false positives
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
    valid = []
    for mid in hf_ids:
        try:
            info = api.model_info(mid)
            if info and getattr(info, "modelId", None):
                valid.append(mid)
        except Exception:
            continue
    return valid


def score_and_decide(row: PaperRow) -> PaperRow:
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

    score += min(row.rationale_hits, 3)

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
    if row.openalex_id:
        return f"oa:{row.openalex_id}"
    d = norm_doi(row.doi)
    if d:
        return f"doi:{d}"
    # fallback fingerprint
    first_author = (row.authors.split(",")[0].strip().lower() if row.authors else "")
    return f"fp:{norm_title(row.title)}|{row.year or ''}|{first_author}"

def merge_rows(a: PaperRow, b: PaperRow) -> PaperRow:
    """Return the better row, merging list fields."""
    # Merge list fields
    a.github_repos = sorted(set(a.github_repos) | set(b.github_repos))
    a.hf_models = sorted(set(a.hf_models) | set(b.hf_models))
    a.hf_models_valid = sorted(set(a.hf_models_valid) | set(b.hf_models_valid))

    # Prefer richer scalar fields
    if not a.doi and b.doi:
        a.doi = b.doi
    if not a.venue and b.venue:
        a.venue = b.venue
    if not a.pdf_url and b.pdf_url:
        a.pdf_url = b.pdf_url
    if not a.landing_page_url and b.landing_page_url:
        a.landing_page_url = b.landing_page_url
    if not a.abstract and b.abstract:
        a.abstract = b.abstract
    if not a.authors and b.authors:
        a.authors = b.authors
    if not a.year and b.year:
        a.year = b.year
    if not a.title and b.title:
        a.title = b.title

    # Rationale: keep max
    a.rationale_hits = max(a.rationale_hits, b.rationale_hits)

    # Re-score after merging
    return score_and_decide(a)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--query",
        action="append",
        default=[],
        help="OpenAlex search string. Use multiple --query flags to run multiple queries.",
    )
    ap.add_argument("--year-from", type=int, default=2021)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--mailto", default=os.getenv("OPENALEX_MAILTO", ""), help="Your email for polite pool")
    ap.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    ap.add_argument("--download-pdfs", action="store_true")
    ap.add_argument("--out-csv", default="11-RECOMMENDATION_EVALUATION/paper_model_3/hf_case_study_candidates_openalex_3.csv")
    ap.add_argument("--out-json", default="11-RECOMMENDATION_EVALUATION/paper_model_3/hf_case_study_candidates_openalex_3.json")
    args = ap.parse_args()

    queries = args.query or [DEFAULT_QUERY]

    mailto = args.mailto.strip() or None
    hf_token = args.hf_token.strip() or None

    candidates = fetch_candidates_openalex(
        search_query=args.query,
        year_from=args.year_from,
        limit=args.limit,
        mailto=mailto,
    )

    all_by_key: Dict[str, PaperRow] = {}

    for q in (args.query or [DEFAULT_QUERY]):
        candidates = fetch_candidates_openalex(
            search_query=q,
            year_from=args.year_from,
            limit=args.limit,
            mailto=mailto,
        )

        for row in candidates:
            text = ""
            if args.download_pdfs and row.pdf_url:
                try:
                    text = safe_get_text_from_pdf_url(row.pdf_url)
                except Exception:
                    text = ""

            scan_text = text if text.strip() else (row.abstract or "")

            gh, hf, rh, doi_found = extract_signals(scan_text)
            row.github_repos = gh
            row.hf_models = hf
            row.rationale_hits = rh
            if not row.doi and doi_found:
                row.doi = doi_found

            row.hf_models_valid = validate_hf_models(row.hf_models, hf_token=hf_token)
            row = score_and_decide(row)

            k = dedupe_key(row)
            if k in all_by_key:
                all_by_key[k] = merge_rows(all_by_key[k], row)
            else:
                all_by_key[k] = row

    results = list(all_by_key.values())


    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.out_csv} and {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    os.environ["HF_TOKEN"]= "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"
    os.environ["OPENALEX_MAILTO"]="ardacanseradali@outlook.com"
    import sys
    sys.argv = [
    sys.argv[0],

    "--query", '("huggingface" OR "hugging face") AND (model OR checkpoint OR backbone) AND (selected OR chosen OR adopted) AND (classification OR detection OR generation OR "question answering")',

    "--query", '(transformers) AND ("fine-tuned" OR finetuned OR "adapted from") AND (because OR due to OR rationale OR motivated) AND (task OR dataset)',

    "--query", '("pre-trained" OR pretrained) AND (model OR backbone) AND (compared OR evaluated OR benchmarked) AND (selected OR best OR outperform*)',

    "--query", '("model hub" OR "model zoo" OR huggingface) AND (selected OR picked OR adopted) AND (task OR application)',

    "--query", '("foundation model" OR "large language model" OR "vision-language model") AND (choose OR selecting OR selection) AND (task OR downstream)',

    "--query", '(benchmark OR leaderboard) AND (model OR checkpoint) AND (selected OR best-performing OR top-performing)',

    "--query", '("parameter count" OR "model size" OR "compute budget" OR latency OR efficiency) AND (model OR backbone) AND (selected OR chosen)',

    "--query", '(BERT OR RoBERTa OR T5 OR GPT OR "sentence transformers") AND (huggingface OR transformers) AND (chosen OR selected) AND (classification OR "question answering" OR retrieval)',

    "--query", '(ResNet OR ViT OR "vision transformer" OR ConvNet) AND (huggingface OR pretrained) AND (selected OR adopted) AND (detection OR segmentation OR classification)',

    "--query", '(diffusers OR "stable diffusion" OR "latent diffusion") AND (checkpoint OR model) AND (selected OR fine-tuned) AND (generation OR synthesis)',

    "--query", '(multilingual OR domain-specific OR biomedical OR legal) AND (model OR checkpoint) AND (selected OR chosen) AND (huggingface OR pretrained)',

    "--query", '("we evaluated" OR "we compared") AND (models OR checkpoints) AND (selected OR chose OR best) AND (task)',

    "--query", '("instruction-tuned" OR "chat model" OR "base model") AND (selected OR chosen) AND (task OR application)',

    "--query", '("huggingface pipeline" OR transformers) AND (applied OR deployed) AND (model OR checkpoint) AND (selected OR chosen)',

    "--query", '("transfer learning" OR "fine-tuning") AND (pretrained model OR backbone) AND (selected OR adopted) AND (task)',

    "--query", '("huggingface hub" OR "model hub") AND (search OR filter OR select) AND (model)',

    "--query", '(ablation OR comparison) AND (models OR backbones) AND (selected OR best) AND (task)',

    "--query", '(real-time OR edge OR deployment) AND (model OR backbone) AND (selected OR chosen) AND (huggingface OR pretrained)',

    "--query", '(embeddings OR retriever OR "sentence transformer") AND (huggingface) AND (selected OR chosen) AND (retrieval OR RAG)',

    "--query", '(checkpoint OR weights) AND (best OR optimal OR selected) AND (fine-tuned OR pretrained) AND (task OR dataset)',

    "--query", '(biomedical OR clinical OR "electronic health record" OR EHR) AND (BioBERT OR PubMedBERT OR "domain-specific" OR huggingface) AND (selected OR chosen OR fine-tuned)',

    "--query", '(legal OR contracts OR "case law" OR legislation) AND (LegalBERT OR pretrained OR huggingface) AND (selected OR chosen OR evaluated)',

    "--query", '(financial OR "earnings call" OR SEC OR "market data") AND (FinBERT OR transformer OR huggingface) AND (selected OR adopted)',

    "--query", '("scientific papers" OR "literature mining" OR "citation analysis") AND (SciBERT OR pretrained OR huggingface) AND (selected OR fine-tuned)',

    "--query", '(chemistry OR molecules OR SMILES OR drug) AND (transformer OR diffusion OR huggingface) AND (selected OR chosen)',

    "--query", '(code OR programming OR software OR "source code") AND (CodeBERT OR CodeT5 OR LLM) AND (selected OR chosen OR evaluated)',

    "--query", '(education OR tutoring OR "intelligent tutoring") AND (LLM OR transformers OR huggingface) AND (selected OR chosen)',

    "--query", '(multilingual OR "low-resource" OR cross-lingual) AND (XLM OR mBERT OR pretrained OR huggingface) AND (selected OR chosen)',

    "--query", '("medical imaging" OR radiology OR pathology) AND (ViT OR CNN OR "pretrained model" OR huggingface) AND (selected OR fine-tuned)',

    "--query", '(satellite OR "remote sensing" OR geospatial) AND (ViT OR pretrained OR foundation model) AND (selected OR chosen)',

    "--query", '(baseline OR "strong baseline") AND (model OR backbone) AND (selected OR chosen) AND (huggingface OR pretrained)',

    "--query", '("state of the art" OR SOTA) AND (model OR checkpoint) AND (selected OR chosen) AND (task)',

    "--query", '("we use" OR "we adopt") AND (pretrained model OR backbone) AND (because OR due to) AND (performance OR efficiency)',

    "--query", '(scalable OR lightweight OR compact) AND (model OR transformer) AND (selected OR chosen) AND (deployment)',

    "--query", '("zero-shot" OR "few-shot") AND (LLM OR transformer) AND (selected OR chosen) AND (task)',

    "--query", '(robust OR robustness) AND (model OR backbone) AND (selected OR chosen) AND (evaluation)',

    "--query", '("general-purpose" OR "domain-adapted") AND (model OR checkpoint) AND (selected OR chosen)',

    "--query", '("open-source" OR license OR "commercial use") AND (model OR checkpoint) AND (selected OR chosen)',

    "--query", '(distilled OR quantized OR pruned) AND (model OR checkpoint) AND (selected OR chosen)',

    "--query", '("long-context" OR "large context window") AND (LLM OR transformer) AND (selected OR chosen)',

    "--query", '(architecture OR backbone) AND (transformer OR CNN OR ViT) AND (selected OR chosen) AND (task)',

    "--query", '("pretraining corpus" OR "training data") AND (model OR checkpoint) AND (selected OR chosen)',

    "--query", '(ablation OR "model study") AND (multiple models OR backbones) AND (selected OR best)',

    "--query", '("off-the-shelf" OR "out-of-the-box") AND (model OR transformer) AND (selected OR chosen)',

    "--query", '(generalization OR "cross-domain") AND (model OR checkpoint) AND (selected OR chosen)',

    "--query", '("few parameters" OR "small model") AND (model OR backbone) AND (selected OR chosen)',

    "--query", '("large model" OR "scaling") AND (LLM OR transformer) AND (selected OR chosen)',

    "--query", '(ensemble OR "model mixture") AND (models OR checkpoints) AND (selected OR best)',

    "--query", '("task-specific" OR "domain-specific") AND (pretrained model OR backbone) AND (selected OR chosen)',

    "--query", '(transferability OR "representation learning") AND (model OR checkpoint) AND (selected OR chosen)',

    "--limit", "1",
]
    main()
