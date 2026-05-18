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
SAMPLE = "M" #CHANGE THIS FOR DIFFERENT TRIALS
EVAL_GLOB = f"eval_{SAMPLE}*.json"            # pattern inside folder
OUT_CSV_PATH = f"8-CRITERIA_SELECTION/hits/case_study_{SAMPLE}.csv"

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
    eval_map[sample_key] = (recs, scores)


def get_recs(sample):
    if sample in eval_map:
        return eval_map[sample][0]  # full list from hits.hits
    return np.nan  # means: eval file not found/mapped for this sample

def get_scores(sample):
    if sample in eval_map:
        return eval_map[sample][1]
    return np.nan

df["recommendations"] = df["sample"].map(get_recs)
df["#rec"] = df["recommendations"].apply(
    lambda x: len(x) if isinstance(x, list) else 0
)
df["display_scores"] = df["sample"].map(get_scores)

# -----------------------
# 4) Compute hits rank
# -----------------------

def compute_hits_models(row):
    target = normalize_model_string(row.get("modelID"))
    recs = row.get("recommendations")

    if target is None or not isinstance(recs, list):
        return np.nan  # no modelID or no eval list

    recs_norm = [normalize_model_string(x) for x in recs]

    # exact match (case-insensitive)
    if target in recs_norm:
        # return the original pretty_id string from recommendations that matched
        idx = recs_norm.index(target)
        return recs[idx]

    return np.nan


def compute_hit_rank(row):
    target = normalize_model_string(row.get("modelID"))
    recs = row.get("recommendations")

    if target is None or not isinstance(recs, list):
        return np.nan

    recs_norm = [normalize_model_string(x) for x in recs]
    try:
        return recs_norm.index(target) + 1  # 1-based
    except ValueError:
        return np.nan


df["hits"] = df.apply(compute_hits_models, axis=1)
df["hit_rank"] = df.apply(compute_hit_rank, axis=1)

# -----------------------
# 5) Accuracy row at top
# -----------------------

# Normalize in approach column (avoid case / whitespace issues)
# df["in approach_norm"] = (df["in approach"].astype(str).str.strip().str.lower())
# SADECE IN APPROACHTA VARSA
# valid_rows = (df["recommendations"].apply(lambda x: isinstance(x, list))& df["modelID"].notna()& (df["in approach"] == "yes"))

# Only rows that have recommendations (i.e., eval exists)
valid_rows = df["recommendations"].apply(lambda x: isinstance(x, list))

total_valid = valid_rows.sum()
correct = df.loc[valid_rows, "hits"].notna().sum()

accuracy = correct / total_valid if total_valid > 0 else 0.0

# Create a summary row that sits at the top.
# We'll store accuracy in a dedicated 'accuracy' column, and also in 'hits' for visibility.
df["accuracy"] = np.nan

summary = {col: np.nan for col in df.columns}
summary["sample"] = "ACCURACY"
summary["accuracy"] = accuracy
summary["hits"] = f"{correct}/{total_valid}"

df_out = pd.concat([pd.DataFrame([summary]), df], ignore_index=True)

# Optional: write lists as JSON strings to keep the CSV clean/readable in Excel
df_out["recommendations"] = df_out["recommendations"].apply(
    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
)
df_out["display_scores"] = df_out["display_scores"].apply(
    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
)

# -----------------------
# 6) Save
# -----------------------
df_out.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8")
print(f"Saved: {OUT_CSV_PATH}")
print(f"Accuracy: {accuracy:.4f} ({correct}/{total_valid})")
