import json
import re
import os
import hashlib
from typing import List, Dict, Optional, Tuple
import requests
import fitz  # PyMuPDF


# with open('11-RECOMMENDATION_EVALUATION/paper_model_2/hf_simple_benchmark_3.json', 'r') as file:
#     RATION_3 = json.load(file)

# with open('11-RECOMMENDATION_EVALUATION/paper_model/hf_simple_benchmark.json', 'r') as file:
#     RATION = json.load(file)

# RATION.extend(RATION_3)
# print(len(RATION))
# RATION_S = RATION[1:2]

with open('11-RECOMMENDATION_EVALUATION/MORE_PAPERS/hf_simple_benchmark_SS.json', 'r') as file:
    RATION = json.load(file)

def safe_filename(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", s.strip())
    return s[:max_len].strip("_") or "paper"


def download_pdf(url: str, out_dir: str = "pdf_SS_cache", timeout: int = 30) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    # stable name based on url hash
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    path = os.path.join(out_dir, f"{h}.pdf")

    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        return path

    headers = {
        "User-Agent": "Mozilla/5.0 (keyword-snippet-extractor; +https://example.com)"
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        # crude check
        if "pdf" not in r.headers.get("Content-Type", "").lower() and not url.lower().endswith(".pdf"):
            # might still be a PDF; proceed
            pass
        with open(path, "wb") as f:
            f.write(r.content)
        if os.path.getsize(path) < 10_000:
            return None
        return path
    except Exception:
        return None


def extract_paragraphs_pymupdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number, paragraph_text).
    Paragraphs are formed by splitting text on blank lines.
    """
    doc = fitz.open(pdf_path)
    paras: List[Tuple[int, str]] = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")  # plain text
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Many PDFs break lines; we keep paragraphs by blank lines
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        for p in raw_paras:
            # light cleanup: collapse internal whitespace
            p = re.sub(r"[ \t]+", " ", p)
            p = re.sub(r"\n+", "\n", p).strip()
            if len(p) >= 30:  # ignore tiny fragments
                paras.append((i + 1, p))
    return paras

import re
from itertools import product

STOPWORDS = {"uncased", "cased", "v1", "v2", "v3", "final", "hf", "model", "all"}

SIZE_TOKENS = {"tiny", "small", "base", "medium", "large", "xl", "xxl"}

def normalize_tokens(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\-_/]", "", text)
    return text


def split_model_id(model_id):
    """
    sentence-transformers/all-mpnet-base-v2
    → ["sentence-transformers", "all", "mpnet", "base", "v2"]
    """
    model_id = normalize_tokens(model_id)
    model_id = model_id.split("/")[-1]          # drop org
    tokens = re.split(r"[-_]", model_id)
    return [t for t in tokens if t]


def generate_casing(token):
    return {token, token.upper(), token.capitalize()}


def remove_versions(tokens):
    return [t for t in tokens if not re.match(r"v\d+", t)]


def generate_aliases(hf_model_id):
    tokens = split_model_id(hf_model_id)
    tokens_no_ver = [
    t for t in remove_versions(tokens)
    if t not in STOPWORDS
]

    aliases = set()

    # 1. Architecture family (mpnet, bert, t5, etc)
    if tokens:
        if tokens[0] not in STOPWORDS:
            aliases.add(tokens[0])
            aliases |= generate_casing(tokens[0])

    # 2. Drop prefixes like "all", "sentence", etc
    core_tokens = [t for t in tokens_no_ver if t not in {"all", "sentence", "transformers"}]

    # 3. Generate combinations like:
    for k in range(1, min(4, len(core_tokens)+1)):
        for subset in [core_tokens[:k]]:
            for sep in ["-", " ", "_"]:
                base = sep.join(subset)
                aliases.add(base)
                for cased in product(*[generate_casing(t) for t in subset]):
                    aliases.add(sep.join(cased))

    # 4. Add size-specific versions
    for t in tokens_no_ver:
        if t in SIZE_TOKENS:
            for arch in generate_casing(tokens_no_ver[0]):
                aliases.add(f"{arch}-{t}")
                aliases.add(f"{arch} {t}")

    # 5. Clean up garbage
    final = set()
    for a in aliases:
        if len(a) >= 3 and not a.isdigit():
            final.add(a)

    return sorted(final)

def generate_aliases_batch(hf_model_ids):
    """
    Takes a list of Hugging Face model IDs
    and returns a flat list of all possible paper-style aliases.
    """
    all_aliases = set()

    for model_id in hf_model_ids:
        aliases = generate_aliases(model_id)
        all_aliases.update(aliases)

    return sorted(all_aliases)

import re
import json

def scan_one_paper_for_aliases(paper, aliases, before=1, after=1, max_hits=50):
    """
    Scan a single paper PDF for any of the strings in `aliases`.
    Returns a result dict with snippets + which aliases matched.
    """
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        return {"paper_id": paper.get("paper_id"), "title": paper.get("title",""), "status": "no_pdf_url", "hits": []}

    pdf_path = download_pdf(pdf_url)
    if not pdf_path:
        return {"paper_id": paper.get("paper_id"), "title": paper.get("title",""), "status": "download_failed", "hits": []}

    paras = extract_paragraphs_pymupdf(pdf_path)
    if not paras:
        return {"paper_id": paper.get("paper_id"), "title": paper.get("title",""), "status": "no_text_extracted", "hits": []}

    # compile each alias as a literal, case-insensitive regex
    pats = [(a, re.compile(re.escape(a), re.IGNORECASE)) for a in aliases if a]

    hits = []
    for i, (page, para) in enumerate(paras):
        matched = [a for a, pat in pats if pat.search(para)]
        if not matched:
            continue

        start = max(0, i - before)
        end = min(len(paras), i + after + 1)

        snippet_parts = []
        pages = set()
        for j in range(start, end):
            pages.add(paras[j][0])
            snippet_parts.append(paras[j][1])

        hits.append({
            "pages": sorted(pages),
            "matched_aliases": matched,
            "snippet": "\n\n".join(snippet_parts),
        })

        if len(hits) >= max_hits:
            break

    return {
        "paper_id": paper.get("paper_id") or paper.get("openalex_id"),
        "title": paper.get("title", ""),
        "pdf_url": pdf_url,
        "status": "ok",
        "alias_count": len(aliases),
        "hit_count": len(hits),
        "hits": hits,
    }


def scan_all_papers_using_alias_lists(papers, before=1, after=1):
    results = []
    total_hits = 0

    for paper in papers:
        aliases = generate_aliases_batch(paper.get("hf_models", []))
        res = scan_one_paper_for_aliases(paper, aliases, before=before, after=after)

        # add up totals
        if res.get("status") == "ok":
            total_hits += res.get("hit_count", 0)

        results.append(res)

    return {
        "total_papers": len(papers),
        "total_hits": total_hits,
        "results": results,
    }

def print_hit_summary(scan_result):
    total = 0
    print("\n========== KEYWORD SCAN SUMMARY ==========")
    print(f"Total papers: {scan_result['total_papers']}")
    print(f"Total hits:   {scan_result['total_hits']}")
    print("------------------------------------------")

    for r in scan_result["results"]:
        pid = r.get("paper_id", "UNKNOWN")
        hits = r.get("hit_count", 0)
        if hits > 0:
            total += 1

        print(f"{pid} | {hits:3d} hits |")
        print("------------------------------------------")
    print(" === === === ")
    # print(f"Papers with hits: {total} / {len(scan_result["results"])}")
    print(f"Papers with hits: {total} / {len(scan_result['results'])}")




# ---- run ----
papers = RATION # list of dicts
results = scan_all_papers_using_alias_lists(papers, before=1, after=1)
print_hit_summary(results)

with open("11-RECOMMENDATION_EVALUATION/MORE_PAPERS/alias_hits.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


import re
import json
from typing import Dict, List, Any, Tuple, Set


# --- optional but strongly recommended for PDFs ---
# Treat weird dashes/underscores/spaces/slashes as equivalent.
SEP_CLASS = r"[\s_\-/\u2013\u2014]*"  # space, _, -, /, en dash, em dash

def alias_to_fuzzy_regex(alias: str) -> str:
    """
    Convert a literal alias like 'tiiuae/falcon-40b' into a regex that tolerates
    PDF text separator variations: spaces, underscores, hyphens, slashes, en/em dashes.
    """
    esc = re.escape(alias)
    return (
        esc.replace(r"\ ", SEP_CLASS)
           .replace(r"\-", SEP_CLASS)
           .replace(r"\_", SEP_CLASS)
           .replace(r"\/", SEP_CLASS)
    )


def normalize_for_dedupe(text: str) -> str:
    """
    Normalize snippet text so cosmetically-different extractions dedupe well.
    """
    t = text.lower()
    t = t.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash -> hyphen
    t = re.sub(r"\s+", " ", t)                          # collapse whitespace
    t = re.sub(r"[^\w\s\-]", "", t)                     # drop most punctuation
    return t.strip()


def scan_paper_for_hf_model_alias_hits(
    paper: Dict[str, Any],
    *,
    before: int = 1,
    after: int = 1,
    max_snippets: int = 100,
) -> Dict[str, Any]:
    """
    For one paper:
    - generate aliases per original hf_model
    - scan PDF paragraphs
    - produce unique snippets
    - report which original hf_models matched at least once
    """

    paper_id = paper.get("paper_id") or paper.get("openalex_id")
    title = paper.get("title", "")
    pdf_url = paper.get("pdf_url")
    hf_models = paper.get("hf_models") or []

    if not pdf_url:
        return {
            "paper_id": paper_id,
            "title": title,
            "pdf_url": pdf_url,
            "status": "no_pdf_url",
            "hf_models": hf_models,
            "matched_hf_models": [],
            "snippets": [],
        }

    # Download + extract
    pdf_path = download_pdf(pdf_url)
    if not pdf_path:
        return {
            "paper_id": paper_id,
            "title": title,
            "pdf_url": pdf_url,
            "status": "download_failed",
            "hf_models": hf_models,
            "matched_hf_models": [],
            "snippets": [],
        }

    paras: List[Tuple[int, str]] = extract_paragraphs_pymupdf(pdf_path)
    if not paras:
        return {
            "paper_id": paper_id,
            "title": title,
            "pdf_url": pdf_url,
            "status": "no_text_extracted",
            "hf_models": hf_models,
            "matched_hf_models": [],
            "snippets": [],
        }

    # Build regex patterns grouped by original hf_model
    # patterns_by_model[model_id] = list of compiled regexes for its aliases
    patterns_by_model: Dict[str, List[re.Pattern]] = {}

    for model_id in hf_models:
        # IMPORTANT: your generate_aliases expects a single string model id
        aliases = generate_aliases(model_id)
        # if you want extra robustness, you can also add the raw model_id
        # (sometimes alias gen misses the slash form)
        if model_id not in aliases:
            aliases.append(model_id)

        # compile "fuzzy" patterns
        compiled = []
        for a in aliases:
            a = (a or "").strip()
            if len(a) < 3 or a.isdigit():
                continue
            compiled.append(re.compile(alias_to_fuzzy_regex(a), re.IGNORECASE))

        if compiled:
            patterns_by_model[model_id] = compiled

    if not patterns_by_model:
        return {
            "paper_id": paper_id,
            "title": title,
            "pdf_url": pdf_url,
            "status": "no_hf_models_or_aliases",
            "hf_models": hf_models,
            "matched_hf_models": [],
            "snippets": [],
        }

    matched_hf_models: Set[str] = set()
    snippets = []
    seen_snippet_keys: Set[Tuple[Tuple[int, ...], str]] = set()

    # Scan paragraphs
    for i, (page, para) in enumerate(paras):
        # Determine which ORIGINAL hf_models match this paragraph
        matched_models_here: Set[str] = set()

        for model_id, pats in patterns_by_model.items():
            # If any alias pattern for that model matches, count model once
            if any(p.search(para) for p in pats):
                matched_models_here.add(model_id)

        if not matched_models_here:
            continue

        # Expand to snippet (before/after paragraphs)
        start = max(0, i - before)
        end = min(len(paras), i + after + 1)

        snippet_parts = []
        pages = set()
        for j in range(start, end):
            pages.add(paras[j][0])
            snippet_parts.append(paras[j][1])

        snippet_text = "\n\n".join(snippet_parts).strip()
        norm = normalize_for_dedupe(snippet_text)

        # Deduplicate snippets across the whole paper
        key = (tuple(sorted(pages)), norm[:2000])  # cap for memory
        if key in seen_snippet_keys:
            # If duplicate snippet appears again, still ensure we record any new matched models
            matched_hf_models |= matched_models_here
            continue
        seen_snippet_keys.add(key)

        matched_hf_models |= matched_models_here

        snippets.append({
            "pages": sorted(pages),
            "matched_hf_models": sorted(matched_models_here),
            "snippet": snippet_text,
        })

        if len(snippets) >= max_snippets:
            break

    return {
        "paper_id": paper_id,
        "title": title,
        "pdf_url": pdf_url,
        "status": "ok",
        "hf_models": hf_models,  # original list preserved
        "matched_hf_models": sorted(matched_hf_models),  # only originals that hit
        "snippet_count": len(snippets),
        "snippets": snippets,  # unique snippets only
    }


def scan_all_papers_keep_hf_models_that_hit(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for paper in papers:
        results.append(scan_paper_for_hf_model_alias_hits(paper, before=1, after=1))
    return results


# ---- run + save ----
papers = RATION
results = scan_all_papers_keep_hf_models_that_hit(papers)
with open("11-RECOMMENDATION_EVALUATION/MORE_PAPERS/alias_hits_2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(len(results), "papers scanned")

import json
from typing import Any, Dict, List, Tuple, Set

import re
import json
from typing import Any, Dict, List, Tuple, Set


def export_matched_papers_summary_json_with_aliases(
    papers: List[Dict[str, Any]],
    out_path: str = "matched_papers_summary.json",
    *,
    before: int = 1,
    after: int = 1,
    max_snippets: int = 100
) -> List[Dict[str, Any]]:
    """
    Writes a JSON file containing ONLY papers that have at least one matched (deduped) snippet.

    Output schema per paper:
      {
        "paper_id": ...,
        "title": ...,
        "pdf_url": ...,
        "snippet_count": <int>,
        "snippet_pages": [<page int>, ...],          # unique pages across all snippets
        "matched_hf_models": ["org/model", ...],     # unique originals that matched
        "matched_aliases": ["alias1", "alias2", ...] # unique matched aliases, lowercased
      }

    Requires these functions to exist:
      - download_pdf(url) -> pdf_path or None
      - extract_paragraphs_pymupdf(pdf_path) -> List[(page_num, paragraph_text)]
      - generate_aliases(hf_model_id: str) -> List[str]
      - alias_to_fuzzy_regex(alias: str) -> str
      - normalize_for_dedupe(text: str) -> str
    """

    matched_summaries: List[Dict[str, Any]] = []

    for paper in papers:
        paper_id = paper.get("paper_id") or paper.get("openalex_id")
        title = paper.get("title", "")
        pdf_url = paper.get("pdf_url")
        hf_models = paper.get("hf_models") or []

        if not pdf_url or not hf_models:
            continue

        pdf_path = download_pdf(pdf_url)
        if not pdf_path:
            continue

        paras: List[Tuple[int, str]] = extract_paragraphs_pymupdf(pdf_path)
        if not paras:
            continue

        # Compile alias patterns per original hf_model, keeping alias text for reporting
        # patterns_by_model[model_id] = [(alias_lower, compiled_regex), ...]
        patterns_by_model: Dict[str, List[Tuple[str, re.Pattern]]] = {}

        for model_id in hf_models:
            aliases = generate_aliases(model_id)  # your function (single model id)
            if model_id not in aliases:
                aliases.append(model_id)  # ensure original is considered too

            compiled_pairs: List[Tuple[str, re.Pattern]] = []
            seen_alias_lower: Set[str] = set()

            for a in aliases:
                a = (a or "").strip()
                if len(a) < 3 or a.isdigit():
                    continue

                a_lower = a.lower()
                if a_lower in seen_alias_lower:
                    continue  # dedupe casing variants here; we keep lowercase as canonical
                seen_alias_lower.add(a_lower)

                compiled_pairs.append(
                    (a_lower, re.compile(alias_to_fuzzy_regex(a), re.IGNORECASE))
                )

            if compiled_pairs:
                patterns_by_model[model_id] = compiled_pairs

        if not patterns_by_model:
            continue

        matched_models: Set[str] = set()
        matched_aliases_lower: Set[str] = set()

        snippet_pages: Set[int] = set()
        seen_snippets: Set[Tuple[Tuple[int, ...], str]] = set()
        snippet_count = 0

        for i, (page, para) in enumerate(paras):
            matched_here_models: Set[str] = set()
            matched_here_aliases: Set[str] = set()

            # Find which ORIGINAL models + which aliases matched this paragraph
            for model_id, alias_pat_pairs in patterns_by_model.items():
                model_matched = False
                for alias_lower, pat in alias_pat_pairs:
                    if pat.search(para):
                        model_matched = True
                        matched_here_aliases.add(alias_lower)  # store canonical lowercase
                if model_matched:
                    matched_here_models.add(model_id)

            if not matched_here_models:
                continue

            # Build snippet window (before/after paragraphs)
            start = max(0, i - before)
            end = min(len(paras), i + after + 1)

            parts = []
            pages = set()
            for j in range(start, end):
                pages.add(paras[j][0])
                parts.append(paras[j][1])

            snippet_text = "\n\n".join(parts).strip()
            norm = normalize_for_dedupe(snippet_text)

            # Deduplicate snippets
            key = (tuple(sorted(pages)), norm[:2000])
            if key in seen_snippets:
                # still accumulate global match sets
                matched_models |= matched_here_models
                matched_aliases_lower |= matched_here_aliases
                snippet_pages |= pages
                continue

            seen_snippets.add(key)
            snippet_count += 1

            matched_models |= matched_here_models
            matched_aliases_lower |= matched_here_aliases
            snippet_pages |= pages

            if snippet_count >= max_snippets:
                break

        # Only keep papers that matched at least one UNIQUE snippet
        if snippet_count > 0:
            matched_summaries.append({
                "paper_id": paper_id,
                "title": title,
                "pdf_url": pdf_url,
                "snippet_count": snippet_count,
                "snippet_pages": sorted(snippet_pages),
                "matched_hf_models": sorted(matched_models),
                "matched_aliases": sorted(matched_aliases_lower),
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(matched_summaries, f, indent=2, ensure_ascii=False)

    return matched_summaries


summaries = export_matched_papers_summary_json_with_aliases(RATION, "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/matched_papers_summary.json")
print(len(summaries), "papers had matches")
