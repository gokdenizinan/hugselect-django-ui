import re

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
RATIONALE_CUES = re.compile(r'\b(because|due to|to handle|to address|as it|since|therefore|we choose|we selected|we adopt)\b', re.IGNORECASE)

def extract_rationale_snippets(text: str, hf_models: list[str], max_snippets: int = 5) -> list[dict]:
    """
    Returns list of snippets with evidence sentences near HF model mentions or HF usage keywords.
    """
    if not text.strip():
        return []

    sentences = SENT_SPLIT.split(text)
    snippets = []

    # Build mention regexes
    mention_terms = set(hf_models)
    # also watch for HF usage even if model id missing
    extra_terms = ["huggingface", "hugging face", "from_pretrained", "transformers", "diffusers", "hf_hub_download", "snapshot_download"]
    mention_terms |= set(extra_terms)

    # Precompile patterns (escape model ids)
    patterns = [re.compile(re.escape(t), re.IGNORECASE) for t in mention_terms if t]

    for i, s in enumerate(sentences):
        if any(p.search(s) for p in patterns):
            # take a window of sentences around it
            start = max(0, i - 2)
            end = min(len(sentences), i + 3)
            window = " ".join(sentences[start:end]).strip()

            # prioritize windows that actually look like rationale
            score = 1
            if RATIONALE_CUES.search(window):
                score += 2

            snippets.append({"score": score, "window": window})

    # return best few
    snippets.sort(key=lambda x: x["score"], reverse=True)
    # de-duplicate exact windows
    seen = set()
    out = []
    for sn in snippets:
        if sn["window"] in seen:
            continue
        seen.add(sn["window"])
        out.append(sn)
        if len(out) >= max_snippets:
            break
    return out

import json
import re
from pathlib import Path


# --- Filtering: remove obvious false positives from hf_models list ---
# A strict rule: accept only "org/model" with no dunder tokens and no "."
BAD_TOKENS = {"__dict__", "__getitem__", "__len__", "__path__", "__version__", "__name__"}
def is_plausible_hf_id(s: str) -> bool:
    if not s or "/" not in s:
        return False
    org, model = s.split("/", 1)
    if not org or not model:
        return False
    # reject python dunder-like artifacts
    if "__" in org or "__" in model:
        return False
    if any(t in s for t in BAD_TOKENS):
        return False
    # reject weird things like "args./dict__"
    if "." in org or "./" in s:
        return False
    # very small org/model strings are usually junk
    if len(org) < 2 or len(model) < 2:
        return False
    return True


def concat_repo_evidence_text(paper: dict) -> str:
    """
    Build a text blob from repo scan evidence that can contain rationale.
    Ideally you would use README/docs rationale evidence if you captured it.
    Here we fallback to hf_evidence snippets and library evidence snippets.
    """
    parts = []
    for scan in paper.get("repo_scans", []) or []:
        # If later you add 'rationale_evidence' to scan, include it here first:
        for ev in scan.get("rationale_evidence", []) or []:
            parts.append(ev.get("snippet", ""))

        # fallback: hf_evidence (often contains model loading or model references)
        for ev in scan.get("hf_evidence", [])[:30]:
            parts.append(ev.get("snippet", ""))

        # fallback: evidence of model load calls (if any)
        for ev in scan.get("model_load_evidence", [])[:30]:
            parts.append(ev.get("snippet", ""))

    return "\n".join(p for p in parts if p)


import json
import re
from pathlib import Path

# --- Adjust these file names to yours ---
# IN_JSON = "11-RECOMMENDATION_EVALUATION/paper_model_2/hf_case_study_candidates_openalex_with_repo_scans_and_rationale_3.json"
# OUT_JSON = "11-RECOMMENDATION_EVALUATION/paper_model_2/hf_simple_benchmark_3.json"

IN_JSON = "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_all_RepoScans_Rationale.json"
OUT_JSON = "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/hf_simple_benchmark_SS.json"
with open("1-MODEL_FILTERING/N_sorted_model_likes_P9.json", "r", encoding="utf-8") as f:
    model_names_dict = json.load(f)

VALID_HF_MODELS = set(model_names_dict.keys())

# Filter out obvious junk HF IDs (keeps it strict)

def is_valid_hf_model(s: str) -> bool:
    if not s:
        return False
    return s in VALID_HF_MODELS



def save_if_model_selected():
    data = json.loads(Path(IN_JSON).read_text(encoding="utf-8"))

    simplified = []
    for paper in data:
        title = paper.get("title", "")
        pdf_url = paper.get("pdf_url", "")
        year = paper.get("year", None)
        doi = paper.get("doi", "")
        paper_id = paper.get("openalex_id") or paper.get("s2_paper_id") or paper.get("id") or ""

        github_repos = paper.get("github_repos") or []

        # --- Collect HF models from all places ---
        # --- Collect HF models from all places ---
        hf_models = set()

        # 1) paper-level
        for m in (paper.get("hf_models") or []):
            if is_valid_hf_model(m):
                hf_models.add(m)

        # 2) repo scans (strongest)
        for scan in (paper.get("repo_scans") or []):
            for m in (scan.get("hf_models") or []):
                if is_valid_hf_model(m):
                    hf_models.add(m)

        # 3) rationale-level
        rationale_obj = paper.get("rationale") or {}
        for m in (rationale_obj.get("hf_models_used") or []):
            if is_valid_hf_model(m):
                hf_models.add(m)

        hf_models = sorted(hf_models)


        # --- Rationale snippets (if present) ---
        snippets_in = (rationale_obj.get("snippets") or [])
        rationale_out = []
        for sn in snippets_in:
            text = sn.get("window") or sn.get("text") or ""
            if not text.strip():
                continue
            rationale_out.append({
                "source": sn.get("source", ""),
                "score": sn.get("score", None),
                "text": text.strip()
            })

        # Only keep entries where we actually found HF models
        if not hf_models:
            continue

        simplified.append({
            "paper_id": paper_id,
            "title": title,
            "pdf_url": pdf_url,
            "doi": doi,
            "year": year,
            "github_repos": github_repos,
            "hf_models": hf_models,
            "rationale": rationale_out  # may be empty []
        })

    Path(OUT_JSON).write_text(json.dumps(simplified, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(simplified)} items to: {OUT_JSON}")




def main():
    # in_path = Path("11-RECOMMENDATION_EVALUATION/paper_model_2/hf_case_study_candidates_openalex_with_repo_scans_3.json")
    # out_path = Path("11-RECOMMENDATION_EVALUATION/paper_model_2/hf_case_study_candidates_openalex_with_repo_scans_and_rationale_3.json")
    in_path = Path("11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_all_RepoScans.json")
    out_path = Path("11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_all_RepoScans_Rationale.json")
    with open("1-MODEL_FILTERING/N_sorted_model_likes_P9.json",  "r") as f:
        model_names_dict = json.load(f)
    VALID_HF_MODELS = list(model_names_dict.keys())

    data = json.loads(in_path.read_text(encoding="utf-8"))

    for idx, paper in enumerate(data, 1):
        title = paper.get("title", "")[:80]

        # collect hf models from repo scans
        hf_models_raw = []
        for scan in paper.get("repo_scans", []) or []:
            hf_models_raw.extend(scan.get("hf_models", []) or [])

        hf_models = sorted({
            m.strip()
            for m in hf_models_raw
            if is_plausible_hf_id(m.strip()) and m.strip() in VALID_HF_MODELS
        })



        # normalize & filter
        hf_models = sorted({m.strip() for m in hf_models if is_plausible_hf_id(m.strip())})
        if hf_models:
            print(f"\n[{idx}/{len(data)}] {title}")
            print(f"  HF models found: {len(hf_models)} -> {hf_models}")



        # If no HF models, skip rationale extraction (or keep empty)
        if not hf_models:
            paper["rationale"] = {
                "hf_models_used": [],
                "snippets": [],
                "sources_used": []
            }
            continue

        # 1) Extract from repo evidence text
        repo_text = concat_repo_evidence_text(paper)
        repo_snips = extract_rationale_snippets(repo_text, hf_models, max_snippets=5)
        for s in repo_snips:
            s["source"] = "repo"

        # 2) Optional: extract from PDF text if you saved it
        # If your collector didn’t store pdf_text, you can add it later.
        pdf_text = paper.get("pdf_text", "") or ""
        pdf_snips = extract_rationale_snippets(pdf_text, hf_models, max_snippets=5)
        for s in pdf_snips:
            s["source"] = "pdf"

        # Merge and keep best
        merged = repo_snips + pdf_snips
        merged.sort(key=lambda x: x["score"], reverse=True)
        merged = merged[:5]
        if hf_models:
            print(f"  Rationale snippets found: {len(merged)}")
            for sn in merged:
                print(f"    - ({sn['source']}, score={sn['score']}) {sn['window'][:120]}...")


        paper["rationale"] = {
            "hf_models_used": hf_models,
            "snippets": merged,
            "sources_used": sorted({s["source"] for s in merged})
        }

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
    save_if_model_selected()
