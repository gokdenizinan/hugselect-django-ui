import os
import json
import glob
import pandas as pd
import numpy as np

# -----------------------
# CONFIG (edit these)
# -----------------------
CSV_PATH = "11-RECOMMENDATION_EVALUATION/paper_model_2/snipped_papers_6.csv"      # your input CSV
OUTPUT_JSON_PATH = "11-RECOMMENDATION_EVALUATION/OUTPUT_F.json"      # your OUTPUT json
EVAL_FOLDER = "8-CRITERIA_SELECTION/user_intent/recommendation_output"            # folder containing eval_*.json
SAMPLE = "AAA" #CHANGE THIS FOR DIFFERENT TRIALS
EVAL_GLOB = f"eval_{SAMPLE}*.json"            # pattern inside folder
OUT_CSV_PATH = f"8-CRITERIA_SELECTION/hits/case_study_united_{SAMPLE}.csv"

# If your eval files map by sample name inside the filename, this regex-like rule is used:
# We'll try to find "sample_#" or "paper_#" in the filename and map it to sample_#
# Example: eval_F_sample_1.json -> sample_1
# If not found, we'll fallback to base filename without extension as a key.
# -----------------------

KEEP_COLS = ["sample", "paper", "title", "modelID", "year", "venue"]
ADD_FROM_OUTPUT = ["model_full_name", "in approach", "task", "domain", "user_intent"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_to_paper_key(sample_val: str) -> str:
    """
    sample_1 -> paper_1
    """
    if isinstance(sample_val, str) and sample_val.startswith("sample_"):
        # return "paper_" + sample_val.split("sample_", 1)[1]
        return "sample_" + sample_val.split("sample_", 1)[1]
    return None


def extract_recs_from_eval(eval_data: dict):
    """
    From eval JSON structure, extract (pretty_id list, display_score list)
    """
    hits = eval_data.get("hits", {}).get("hits", [])
    pretty_ids = []
    display_scores = []
    for item in hits:
        pretty_ids.append(item.get("pretty_id"))
        display_scores.append(item.get("display_score"))
    # Clean Nones while keeping alignment (optional):
    # We'll keep Nones so index alignment stays consistent.
    return pretty_ids, display_scores

import os, re

import os, re

def infer_sample_key_from_filename(filename: str):
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"(?:^|[_-])sample[_-]?(\d+)(?:$|[_-])", stem, flags=re.IGNORECASE)
    if m:
        return f"sample_{m.group(1)}"
    m = re.search(r"eval[_-]?[a-zA-Z](\d+)", stem)
    if m:
        return f"sample_{m.group(1)}"

    return None

def normalize_model_string(x):
    """
    Normalize for matching:
    - lower
    - strip whitespace
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return str(x).strip().lower()


# -----------------------
# 1) Load CSV and keep columns
# -----------------------
df = pd.read_csv(
    CSV_PATH,
    sep=";",
    engine="python"
)
df = df.rename(columns={
    "saved_name": "paper",
    "matched_hf_model": "modelID"
})


missing = [c for c in KEEP_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Input CSV is missing required columns: {missing}")

df = df[KEEP_COLS].copy()

# -----------------------
# 2) Load OUTPUT.json and add columns (SAFE)
# -----------------------
# -----------------------
# 2) Load OUTPUT.json and keep only samples present there
# -----------------------
output_data = load_json(OUTPUT_JSON_PATH)

def sample_exists_in_output(sample):
    paper_key = sample_to_paper_key(sample)
    return (
        paper_key is not None
        and isinstance(output_data, dict)
        and paper_key in output_data
    )

df = df[df["sample"].apply(sample_exists_in_output)].copy()

sample_to_fields = {}

for _, row in df.iterrows():
    sample = row["sample"]
    paper_key = sample_to_paper_key(sample)

    fields = {k: None for k in ADD_FROM_OUTPUT}

    paper_obj = output_data.get(paper_key, {})
    if not isinstance(paper_obj, dict):
        paper_obj = {}

    for k in ADD_FROM_OUTPUT:
        fields[k] = paper_obj.get(k)

    sample_to_fields[sample] = fields

for k in ADD_FROM_OUTPUT:
    df[k] = df["sample"].map(lambda s: sample_to_fields.get(s, {}).get(k))

# -----------------------
# 3) Load eval_F*.json files and extract recommendations + display_scores
# -----------------------
eval_paths = sorted(glob.glob(os.path.join(EVAL_FOLDER, EVAL_GLOB)))
print("Found eval files:", len(eval_paths))

eval_map = {}
for p in eval_paths:
    sample_key = infer_sample_key_from_filename(p)
    if sample_key is None:
        continue

    data = load_json(p)
    recs, scores = extract_recs_from_eval(data)
    eval_map[sample_key] = {
        "recommendations": recs,
        "display_scores": scores
    }

# -----------------------
# 3b) Your extra recommendation dicts
# -----------------------
from F_Hits_United import CHATGPT_RECOMMENDATIONS, GEMINI_RECOMMENDATIONS
# -----------------------
# 4) Generic evaluation helper
# -----------------------
def compute_hits_models_from_recs(target_model, recs):
    target = normalize_model_string(target_model)

    if target is None or not isinstance(recs, list):
        return np.nan

    recs_norm = [normalize_model_string(x) for x in recs]
    if target in recs_norm:
        idx = recs_norm.index(target)
        return recs[idx]
    return np.nan


def compute_hit_rank_from_recs(target_model, recs):
    target = normalize_model_string(target_model)

    if target is None or not isinstance(recs, list):
        return np.nan

    recs_norm = [normalize_model_string(x) for x in recs]
    try:
        return recs_norm.index(target) + 1
    except ValueError:
        return np.nan


def add_recommendation_eval_columns(
    df,
    sample_to_recs,
    prefix,
    include_display_scores=False,
    sample_to_scores=None
):
    rec_col = f"{prefix}_recommendations"
    nrec_col = f"{prefix}_#rec"
    hits_col = f"{prefix}_hits"
    rank_col = f"{prefix}_hit_rank"
    acc_col = f"{prefix}_accuracy"

    df[rec_col] = df["sample"].map(sample_to_recs)
    df[nrec_col] = df[rec_col].apply(lambda x: len(x) if isinstance(x, list) else 0)

    if include_display_scores:
        score_col = f"{prefix}_display_scores"
        df[score_col] = df["sample"].map(sample_to_scores if sample_to_scores else {})

    df[hits_col] = df.apply(
        lambda row: compute_hits_models_from_recs(row.get("modelID"), row.get(rec_col)),
        axis=1
    )
    df[rank_col] = df.apply(
        lambda row: compute_hit_rank_from_recs(row.get("modelID"), row.get(rec_col)),
        axis=1
    )

    # valid_rows = df[rec_col].apply(lambda x: isinstance(x, list))
    valid_rows = df[rec_col].apply(lambda x: isinstance(x, list) and len(x) > 0)
    total_valid = valid_rows.sum()
    correct = df.loc[valid_rows, hits_col].notna().sum()
    accuracy = correct / total_valid if total_valid > 0 else 0.0

    df[acc_col] = np.nan

    return df, {
        "accuracy": accuracy,
        "correct": correct,
        "total_valid": total_valid,
        "rec_col": rec_col,
        "hits_col": hits_col,
        "rank_col": rank_col,
        "acc_col": acc_col
    }

# -----------------------
# 5) Add evaluations for each source
# -----------------------

# Existing eval JSON recommendations
eval_recommendations = {
    sample: payload["recommendations"]
    for sample, payload in eval_map.items()
}
eval_scores = {
    sample: payload["display_scores"]
    for sample, payload in eval_map.items()
}

df, base_stats = add_recommendation_eval_columns(
    df,
    sample_to_recs=eval_recommendations,
    prefix="base",
    include_display_scores=True,
    sample_to_scores=eval_scores
)

# ChatGPT recommendations
df, chatgpt_stats = add_recommendation_eval_columns(
    df,
    sample_to_recs=CHATGPT_RECOMMENDATIONS,
    prefix="chatgpt"
)

# Gemini recommendations
df, gemini_stats = add_recommendation_eval_columns(
    df,
    sample_to_recs=GEMINI_RECOMMENDATIONS,
    prefix="gemini"
)

# -----------------------
# 6) Summary row at top
# -----------------------
summary = {col: np.nan for col in df.columns}
summary["sample"] = "ACCURACY"

summary["base_accuracy"] = base_stats["accuracy"]
summary["base_hits"] = f'{base_stats["correct"]}/{base_stats["total_valid"]}'

summary["chatgpt_accuracy"] = chatgpt_stats["accuracy"]
summary["chatgpt_hits"] = f'{chatgpt_stats["correct"]}/{chatgpt_stats["total_valid"]}'

summary["gemini_accuracy"] = gemini_stats["accuracy"]
summary["gemini_hits"] = f'{gemini_stats["correct"]}/{gemini_stats["total_valid"]}'

df_out = pd.concat([pd.DataFrame([summary]), df], ignore_index=True)

# Optional: stringify list columns for CSV output
list_like_cols = [
    "base_recommendations",
    "base_display_scores",
    "chatgpt_recommendations",
    "gemini_recommendations",
]

for col in list_like_cols:
    if col in df_out.columns:
        df_out[col] = df_out[col].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
        )

# -----------------------
# 7) Save
# -----------------------
df_out.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8")
print(f"Saved: {OUT_CSV_PATH}")
print(f'Base accuracy: {base_stats["accuracy"]:.4f} ({base_stats["correct"]}/{base_stats["total_valid"]})')
print(f'ChatGPT accuracy: {chatgpt_stats["accuracy"]:.4f} ({chatgpt_stats["correct"]}/{chatgpt_stats["total_valid"]})')
print(f'Gemini accuracy: {gemini_stats["accuracy"]:.4f} ({gemini_stats["correct"]}/{gemini_stats["total_valid"]})')