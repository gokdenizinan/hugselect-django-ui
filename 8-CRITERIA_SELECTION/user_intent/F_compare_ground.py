import json
import os
import glob
from typing import Dict, Any, List, Tuple, Optional

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_model_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.removesuffix(".json")
    s = s.replace("__", "/")
    return s


def extract_hits_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Supports both:
      A) {"hits": {"hits": [...]}}  (Elasticsearch)
      B) {"hits": [...]}           (flattened)
    """
    hits = data.get("hits")

    # Case A: ES style
    if isinstance(hits, dict):
        inner = hits.get("hits")
        return inner if isinstance(inner, list) else []

    # Case B: flattened
    if isinstance(hits, list):
        return hits

    return []

def extract_hit_model_ids(hit: Dict[str, Any]) -> List[str]:
    """
    Supports both:
      - ES: hit["_id"], hit["_source"]["modelID"]
      - flattened: hit["id"], hit["pretty_id"]
    """
    out = []

    src = hit.get("_source") or {}
    if isinstance(src, dict):
        mid = src.get("modelID")
        if isinstance(mid, str) and mid.strip():
            out.append(mid)

    for k in ("_id", "id", "pretty_id"):
        v = hit.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v)

    return out


def parse_ground_truth_models(gt_value: str) -> set:
    """
    Parses ground truth value into a set of normalized model names.
    Supports multiple models separated by ';'
    """
    if not gt_value:
        return set()

    models = [
        normalize_model_name(m)
        for m in gt_value.split(";")
        if m.strip()
    ]
    return set(models)

def check_file_match(
    file_path: str,
    ground_truth: Dict[str, str],
    key_from_filename: str,
    match_top_k: Optional[int] = None,
):
    """
    Returns:
      (is_match, query_key, gt_models, hit_models_checked)
    """
    gt_raw = ground_truth.get(key_from_filename)
    if gt_raw is None:
        return (False, key_from_filename, None, [])

    gt_models = parse_ground_truth_models(gt_raw)
    if not gt_models:
        return (False, key_from_filename, gt_models, [])

    data = load_json(file_path)
    hits = extract_hits_list(data)

    if match_top_k is not None:
        hits = hits[:match_top_k]

    if not isinstance(hits, list):
        hits = []

    checked = []

    for hit in hits:
        if not isinstance(hit, dict):
            continue

        candidates = extract_hit_model_ids(hit)
        for c in candidates:
            c_norm = normalize_model_name(c)
            checked.append(c_norm)

            if c_norm in gt_models:
                return (True, key_from_filename, gt_models, checked)

    return (False, key_from_filename, gt_models, checked)


def compute_accuracy(
    folder: str,
    ground_truth: Dict[str, str],
    pattern: str = "eval_B*.json",
    top_k: Optional[int] = None,
    require_gt: bool = True,
) -> Dict[str, Any]:
    """
    Scans folder for eval_*.json, matches per file, computes accuracy.

    Args:
      folder: directory containing eval_*.json
      ground_truth: dict like {"A1": "hkunlp/instructor-large", ...}
      pattern: glob pattern for eval files
      top_k: if set, only check first K hits
      require_gt: if True, files without ground truth are skipped (not counted)

    Returns a summary dict with accuracy and per-file results.
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))

    results = []
    matched = 0
    counted = 0
    skipped_no_gt = 0

    for p in paths:
        base = os.path.basename(p)
        # "eval_A1.json" -> "A1"
        key = base[len("eval_A") : -len(".json")] if base.startswith("eval_") and base.endswith(".json") else base
        key = "A"+key  

        is_match, qkey, gt_model, checked = check_file_match(
            p, ground_truth, key_from_filename=key, match_top_k=top_k
        )

        if gt_model is None:
            skipped_no_gt += 1
            if require_gt:
                continue
            # If not requiring GT, count as incorrect
            counted += 1
            results.append(
                {"file": base, "key": qkey, "gt": None, "match": False, "checked": checked}
            )
            continue

        counted += 1
        matched += int(is_match)
        results.append(
            {"file": base, "key": qkey, "gt": gt_model, "match": is_match, "checked": checked}
        )

    accuracy = (matched / counted) if counted > 0 else 0.0
    return {
        "folder": folder,
        "pattern": pattern,
        "top_k": top_k,
        "counted": counted,
        "matched": matched,
        "accuracy": accuracy,
        "skipped_no_gt": skipped_no_gt,
        "results": results,
    }

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    folder = "8-CRITERIA_SELECTION/user_intent/recommendation_output"

    # # Example ground truth dict:
    # with open("8-CRITERIA_SELECTION/user_intent/ground_truth.json", "r", encoding="utf-8") as f:
    #     ground_truth = json.load(f)
    with open("11-RECOMMENDATION_EVALUATION/OUTPUT_F.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    ground_truth = {
    f"A{i}": v["model_full_name"]
    for i, v in enumerate(
        (value for value in data.values() if value is not None),
        start=1
    )
}
    summary = compute_accuracy(folder, ground_truth, pattern=  "eval_H*.json", top_k=None, require_gt=True)

    print(f"\nCounted files: {summary['counted']}")
    print(f"Matched: {summary['matched']}")
    print(f"Accuracy: {summary['accuracy']:.2f}")

    # Optional: print mismatches
    matches = [r for r in summary["results"] if r["gt"] is not None and r["match"]]
    mismatches = [r for r in summary["results"] if r["gt"] is not None and not r["match"]]

    if matches:
        print("\nMatches:")
        for r in matches[:50]:
            print(f"- {r['file']} (key={r['key']}): gt={r['gt']}, \nlength checked= {len(r['checked'])}, checked_first={r['checked'][:5]}")
    if mismatches:
        print("\nMismatches:")
        for r in mismatches[:50]:
            print(f"- {r['file']} (key={r['key']}): gt={r['gt']}, \nlength checked= {len(r['checked'])}, checked_first={r['checked'][:5]}")

