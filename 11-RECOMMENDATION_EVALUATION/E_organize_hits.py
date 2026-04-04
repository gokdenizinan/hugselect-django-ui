#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    # pypdf is the modern maintained fork of PyPDF2
    from pypdf import PdfReader, PdfWriter
    from pypdf.errors import PdfReadError, PdfStreamError
except ImportError:
    raise SystemExit(
        "Missing dependency: pypdf\n"
        "Install with: pip install pypdf"
    )


def _hash_hex(s: str, algo: str) -> str:
    h = hashlib.new(algo)
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def _candidate_cache_names(pdf_url: str, paper_id: str) -> List[str]:
    """
    Generate a list of plausible cache filenames from common hashing schemes.
    Your cache looks like hex.pdf (no prefix). We'll try:
      - md5(url) full + first 16
      - sha1(url) full + first 16
      - sha256(url) full + first 16
      - same for normalized_url
      - same for paper_id
    """
    def normalize(u: str) -> str:
        u = u.strip()
        u = u.split("#", 1)[0]
        return u

    url = pdf_url.strip()
    nurl = normalize(pdf_url)
    pid = (paper_id or "").strip()

    algos = ["md5", "sha1", "sha256"]
    bases = [url, nurl]
    if pid:
        bases.append(pid)

    cands: List[str] = []
    for base in bases:
        for algo in algos:
            hx = _hash_hex(base, algo)
            cands.append(f"{hx}.pdf")
            cands.append(f"{hx[:16]}.pdf")
            cands.append(f"{hx[:32]}.pdf")

    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def find_cached_pdf(cache_dir: Path, pdf_url: str, paper_id: str) -> Optional[Path]:
    """
    Try to locate the cached PDF file in cache_dir.
    """
    for name in _candidate_cache_names(pdf_url, paper_id):
        p = cache_dir / name
        if p.exists() and p.is_file():
            return p
    return None


def load_anan_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load ANAN JSON and build a lookup table.

    Supported formats:
    - a list of dicts
    - a dict containing a list under one of: papers/items/data/results
    - fallback: dict values that are records

    We index by:
    - paper_id
    - pdf_url
    - title
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = None
        for key in ["papers", "items", "data", "results"]:
            if isinstance(raw.get(key), list):
                items = raw[key]
                break
        if items is None:
            items = [v for v in raw.values() if isinstance(v, dict)]
    else:
        raise SystemExit("ANAN JSON must be a list or a dict containing paper records.")

    lookup: Dict[str, Dict[str, Any]] = {}

    for rec in items:
        if not isinstance(rec, dict):
            continue

        keys = [
            str(rec.get("paper_id", "")).strip(),
            str(rec.get("pdf_url", "")).strip(),
            str(rec.get("title", "")).strip(),
        ]

        for k in keys:
            if k:
                lookup[k] = rec

    return lookup


def get_year_venue(item: Dict[str, Any], anan_lookup: Dict[str, Dict[str, Any]]) -> Tuple[Any, Any]:
    """
    Prefer values already in item; otherwise look them up from ANAN using:
    paper_id -> pdf_url -> title
    """
    year = item.get("year", "")
    venue = item.get("venue", "")

    if year and venue:
        return year, venue

    candidates = [
        str(item.get("paper_id", "")).strip(),
        str(item.get("pdf_url", "")).strip(),
        str(item.get("title", "")).strip(),
    ]

    rec = None
    for key in candidates:
        if key and key in anan_lookup:
            rec = anan_lookup[key]
            break

    if rec:
        year = year or rec.get("year", "")
        venue = venue or rec.get("venue", "")

    return year, venue


def is_probably_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        return header.startswith(b"%PDF-")
    except OSError:
        return False


def select_relevant_pages(
    snippet_pages: Any,
    total_pages: int,
    max_pages: int = 15,
    context: int = 1,
) -> List[int]:
    """
    Prefer snippet_pages; add +/- context around them.
    If still > max_pages, keep snippet pages first, then nearest context pages.
    Returns sorted unique 1-indexed pages within [1, total_pages].
    """
    seeds: List[int] = []
    if snippet_pages:
        for p in snippet_pages:
            try:
                seeds.append(int(p))
            except (TypeError, ValueError):
                pass

    seeds = sorted({p for p in seeds if 1 <= p <= total_pages})
    if not seeds:
        return []

    candidates = set(seeds)
    if context > 0:
        for p in seeds:
            for d in range(1, context + 1):
                if 1 <= p - d <= total_pages:
                    candidates.add(p - d)
                if 1 <= p + d <= total_pages:
                    candidates.add(p + d)

    if len(candidates) <= max_pages:
        return sorted(candidates)

    seed_set = set(seeds)

    def score(page: int) -> tuple:
        is_seed = 0 if page in seed_set else 1
        dist = min(abs(page - s) for s in seeds)
        median = seeds[len(seeds) // 2]
        med_dist = abs(page - median)
        return (is_seed, dist, med_dist, page)

    ranked = sorted(candidates, key=score)
    chosen = sorted(ranked[:max_pages])
    return chosen


def snip_pdf(input_pdf: Path, output_pdf: Path, pages_1_indexed: List[int]) -> int:
    if not is_probably_pdf(input_pdf):
        raise ValueError(f"Not a real PDF (bad header): {input_pdf}")

    try:
        reader = PdfReader(str(input_pdf), strict=False)
        total_pages = len(reader.pages)
    except (PdfReadError, PdfStreamError, Exception) as e:
        raise ValueError(f"Unreadable/corrupt PDF: {input_pdf} :: {e}")

    selected = select_relevant_pages(
        snippet_pages=pages_1_indexed,
        total_pages=total_pages,
        max_pages=15,
        context=1,
    )

    writer = PdfWriter()
    for p in selected:
        writer.add_page(reader.pages[p - 1])

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdf, "wb") as f:
        writer.write(f)

    return len(selected)


def main() -> None:
    ap = argparse.ArgumentParser(description="Snip cached PDFs by page lists and export a CSV summary.")
    ap.add_argument("--cache_dir", type=str, default="pdf_SS_cache", help="Folder containing cached PDFs.")
    ap.add_argument("--meta_json", type=str, required=True, help="Path to JSON file containing the list of paper dicts.")
    ap.add_argument("--anan_json", type=str, required=True, help="Path to ANAN JSON containing year/venue metadata.")
    ap.add_argument("--out_dir", type=str, default="pdf_snippets", help="Output folder for snipped PDFs.")
    ap.add_argument(
        "--csv_out",
        type=str,
        default="11-RECOMMENDATION_EVALUATION/paper_model_2/snipped_papers.csv",
        help="CSV output path.",
    )
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    meta_path = Path(args.meta_json)
    anan_path = Path(args.anan_json)
    csv_out = Path(args.csv_out)

    if not cache_dir.exists():
        raise SystemExit(f"Cache dir not found: {cache_dir.resolve()}")
    if not meta_path.exists():
        raise SystemExit(f"Metadata JSON not found: {meta_path.resolve()}")
    if not anan_path.exists():
        raise SystemExit(f"ANAN JSON not found: {anan_path.resolve()}")

    data: List[Dict[str, Any]] = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Metadata JSON must be a list of objects.")

    anan_lookup = load_anan_metadata(anan_path)

    rows: List[Dict[str, Any]] = []
    paper_idx = 87      # first saved paper becomes paper_88.pdf
    sample_idx = 164    # first sample becomes sample_165
    missing: List[Tuple[str, str]] = []

    for item in data:
        pdf_url = str(item.get("pdf_url", "")).strip()
        paper_id = str(item.get("paper_id", "")).strip()
        title = str(item.get("title", "")).strip()
        snippet_count = item.get("snippet_count", "")
        snippet_pages = item.get("snippet_pages", []) or []
        matched_models = item.get("matched_hf_models", []) or []
        matched_aliases = item.get("matched_aliases", []) or []

        aliases_str = ";".join(map(str, matched_aliases)) if isinstance(matched_aliases, list) else str(matched_aliases)

        if not isinstance(matched_models, list):
            matched_models = [str(matched_models)] if matched_models else []
        if not isinstance(matched_aliases, list):
            matched_aliases = [str(matched_aliases)] if matched_aliases else []

        if not matched_models:
            missing.append((title or paper_id or pdf_url, "no matched_hf_models"))
            continue

        year, venue = get_year_venue(item, anan_lookup)

        cached = find_cached_pdf(cache_dir, pdf_url, paper_id)
        if cached is None:
            missing.append((title or paper_id or pdf_url, "cache miss (hash not found)"))
            continue

        paper_idx += 1
        saved_name = f"paper_{paper_idx}.pdf"
        out_pdf = out_dir / saved_name

        try:
            kept_pages = snip_pdf(cached, out_pdf, list(snippet_pages))
        except Exception as e:
            missing.append((title or paper_id or pdf_url, f"bad cached pdf: {e}"))
            continue

        for model in matched_models:
            sample_idx += 1
            sample_id = f"sample_{sample_idx}"
            model_str = str(model).strip()

            prompt = (
                f"Model full name: {model_str}\n\n"
                "The given PDF contains a research paper that used this model.\n"
                "Extract the gold-standard model-selection record.\n\n"
                "If the model was used in development of the approach/methodology of the paper, then 'in approach' is yes. "
                "If it was only used in experiments or as a baseline, then 'in approach' is no.\n\n"
                "Return a JSON object with:\n"
                "- model_full_name\n"
                "- in approach: yes/no\n"
                "- task: ≤ 8 words\n"
                "- domain: ≤ 8 words\n"
                "- selection_rationale: ≤ 2 sentences\n"
                "- user_intent: ≤ 2 sentences\n"
                "- evidence: ≤ 4 sentences\n\n"
                "Important rules:\n"
                "1. Do NOT propose or infer any other models.\n"
                "2. Extract the rationale only from how the authors justify this model in the paper.\n"
                "3. If the rationale is not explicitly stated, infer it conservatively from the surrounding text.\n"
                "4. Quote verbatim evidence from the paper.\n"
                "5. Ignore models mentioned only in related work.\n"
                "6. Output strictly valid JSON and nothing else.\n\n"
            )

            rows.append(
                {
                    "sample": sample_id,
                    "saved_name": saved_name,
                    "matched_hf_model": model_str,
                    "matched_aliases": aliases_str,
                    "snippet_count": snippet_count,
                    "title": title,
                    "year": year,
                    "venue": venue,
                    "paper_url": paper_id or pdf_url,
                    "kept_pages": kept_pages,
                    "prompt": prompt,
                    "output": "",
                }
            )

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "saved_name",
        "matched_hf_model",
        "matched_aliases",
        "snippet_count",
        "title",
        "year",
        "venue",
        "paper_url",
        "kept_pages",
        "prompt",
        "output",
    ]

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Done.\n- Snipped PDFs written to: {out_dir.resolve()}\n- CSV written to: {csv_out.resolve()}")

    if missing:
        print("\nMissing / skipped papers:")
        for k, reason in missing[:50]:
            print(f"  - {k} :: {reason}")
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more.")


if __name__ == "__main__":
    import sys
    sys.argv = [
        "snip_cached_pdfs.py",
        "--meta_json", "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/matched_papers_summary.json",
        "--anan_json", "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_all_RepoScans_Rationale.json",
        "--cache_dir", "pdf_SS_cache",
        "--out_dir", "11-RECOMMENDATION_EVALUATION/MORE_PAPERS",
        "--csv_out", "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/snipped_papers_7.csv",
    ]
    main()