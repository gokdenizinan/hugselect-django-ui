import json
import glob
import os
import re
import pandas as pd
import numpy as np
import math

# -----------------------
# CONFIG
# -----------------------
CSV_PATH = "11-RECOMMENDATION_EVALUATION/paper_model_2/snipped_papers_6.csv"
OUTPUT_JSON_PATH = "11-RECOMMENDATION_EVALUATION/OUTPUT_F.json"
EVAL_FOLDER = "8-CRITERIA_SELECTION/user_intent/recommendation_output"
SAMPLE = "M"
EVAL_GLOB = f"eval_{SAMPLE}*.json"
OUT_CSV_PATH = f"8-CRITERIA_SELECTION/hits/case_study_short_{SAMPLE}.csv"

from F_Hits_United import CHATGPT_RECOMMENDATIONS, GEMINI_RECOMMENDATIONS

K = 10

KEEP_COLS = ["sample", "paper", "title", "modelID", "year", "venue"]


# -----------------------
# HELPERS
# -----------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_model_string(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    return str(x).strip().lower()


def extract_recs_from_eval(eval_data: dict):
    hits = eval_data.get("hits", {}).get("hits", [])
    pretty_ids = []
    for item in hits:
        pretty_id = item.get("pretty_id")
        if pretty_id is not None:
            pretty_ids.append(pretty_id)
    return pretty_ids


def infer_sample_key_from_filename(filename: str):
    stem = os.path.splitext(os.path.basename(filename))[0]

    m = re.search(r"(?:^|[_-])sample[_-]?(\d+)(?:$|[_-])", stem, flags=re.IGNORECASE)
    if m:
        return f"sample_{m.group(1)}"

    m = re.search(r"eval[_-]?[a-zA-Z](\d+)", stem)
    if m:
        return f"sample_{m.group(1)}"

    return None


def find_rank(target_model, recs):
    target = normalize_model_string(target_model)
    if target is None or not isinstance(recs, list):
        return np.nan

    recs_norm = [normalize_model_string(x) for x in recs]
    try:
        return recs_norm.index(target) + 1  # 1-based
    except ValueError:
        return np.nan


def precision_at_k(rank, k=10):
    # single relevant item
    if pd.isna(rank) or rank > k:
        return 0.0
    return 1.0 / k


def recall_at_k(rank, k=10):
    # single relevant item
    if pd.isna(rank) or rank > k:
        return 0.0
    return 1.0


def ndcg_at_k(rank, k=10):
    # single relevant item, ideal DCG = 1
    if pd.isna(rank) or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def average_precision(rank):
    # single relevant item
    if pd.isna(rank):
        return 0.0
    return 1.0 / rank


def accuracy_from_rank(rank):
    # exact hit anywhere in returned list
    return 0.0 if pd.isna(rank) else 1.0


def coverage_at_k(rank, k=10):
    # same as hit rate in top-k
    return 0.0 if pd.isna(rank) or rank > k else 1.0


def overlap_with_pipeline(source_recs, pipeline_recs):
    """
    Percentage of source recommendations that also appear in pipeline recommendations
    for the same sample.
    overlap = |source ∩ pipeline| / |source|
    """
    if not isinstance(source_recs, list) or len(source_recs) == 0:
        return np.nan
    if not isinstance(pipeline_recs, list):
        return np.nan

    source_set = {normalize_model_string(x) for x in source_recs if normalize_model_string(x) is not None}
    pipeline_set = {normalize_model_string(x) for x in pipeline_recs if normalize_model_string(x) is not None}

    if len(source_set) == 0:
        return np.nan

    return len(source_set & pipeline_set) / len(source_set)


def summarize_source(df, rec_col, label, k=10):
    # valid_mask = df[rec_col].apply(lambda x: isinstance(x, list))
    
    eval_df = df[df[rec_col].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    # eval_df = df.loc[valid_mask].copy()
    print(f"{label}: evaluating {len(eval_df)} samples")
    if len(eval_df) == 0:
        return {
            "source": label,
            "evaluated_samples": 0,
            "accuracy": 0.0,
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            "map": 0.0,
            f"coverage@{k}": 0.0,
            "overlap_with_pipeline": np.nan if label != "pipeline" else 1.0,
        }

    eval_df["rank"] = eval_df.apply(lambda row: find_rank(row["modelID"], row[rec_col]), axis=1)
    eval_df["accuracy_metric"] = eval_df["rank"].apply(accuracy_from_rank)
    eval_df[f"precision@{k}_metric"] = eval_df["rank"].apply(lambda r: precision_at_k(r, k))
    eval_df[f"recall@{k}_metric"] = eval_df["rank"].apply(lambda r: recall_at_k(r, k))
    eval_df[f"ndcg@{k}_metric"] = eval_df["rank"].apply(lambda r: ndcg_at_k(r, k))
    eval_df["ap_metric"] = eval_df["rank"].apply(average_precision)
    eval_df[f"coverage@{k}_metric"] = eval_df["rank"].apply(lambda r: coverage_at_k(r, k))

    if label == "pipeline":
        overlap_value = 1.0
    else:
        eval_df["overlap_metric"] = eval_df.apply(
            lambda row: overlap_with_pipeline(row[rec_col], row["pipeline_recommendations"]),
            axis=1
        )
        overlap_value = eval_df["overlap_metric"].dropna().mean()
        if pd.isna(overlap_value):
            overlap_value = np.nan

    return {
        "source": label,
        "evaluated_samples": int(len(eval_df)),
        "accuracy": round(eval_df["accuracy_metric"].mean(), 4),
        f"precision@{k}": round(eval_df[f"precision@{k}_metric"].mean(), 4),
        f"recall@{k}": round(eval_df[f"recall@{k}_metric"].mean(), 4),
        f"ndcg@{k}": round(eval_df[f"ndcg@{k}_metric"].mean(), 4),
        "map": round(eval_df["ap_metric"].mean(), 4),
        f"coverage@{k}": round(eval_df[f"coverage@{k}_metric"].mean(), 4),
        "overlap_with_pipeline": round(overlap_value, 4) if not pd.isna(overlap_value) else np.nan,
    }


# -----------------------
# 1) LOAD BASE CSV
# -----------------------
df = pd.read_csv(CSV_PATH, sep=";", engine="python")
filter_data = load_json(OUTPUT_JSON_PATH)

# -----------------------
# FILTER USING OUTPUT JSON
# -----------------------
filter_data = load_json(OUTPUT_JSON_PATH)

# valid sample keys
valid_samples = set(filter_data.keys())

print("Samples allowed by filter:", len(valid_samples))

df = df.rename(columns={
    "saved_name": "paper",
    "matched_hf_model": "modelID"
})

missing = [c for c in KEEP_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Input CSV is missing required columns: {missing}")

df = df[KEEP_COLS].copy()

# keep only rows whose sample exists in OUTPUT_JSON
df = df[df["sample"].isin(valid_samples)].copy()

print("Rows after filtering:", len(df))




# -----------------------
# 2) LOAD PIPELINE RECOMMENDATIONS
# -----------------------
eval_paths = sorted(glob.glob(os.path.join(EVAL_FOLDER, EVAL_GLOB)))
print("Found eval files:", len(eval_paths))

PIPELINE_RECOMMENDATIONS = {}
for p in eval_paths:
    sample_key = infer_sample_key_from_filename(p)
    if sample_key is None:
        continue

    data = load_json(p)
    PIPELINE_RECOMMENDATIONS[sample_key] = extract_recs_from_eval(data)


# -----------------------
# 3) ATTACH RECOMMENDATION LISTS
# -----------------------
df["pipeline_recommendations"] = df["sample"].map(PIPELINE_RECOMMENDATIONS)
df["chatgpt_recommendations"] = df["sample"].map(CHATGPT_RECOMMENDATIONS)
df["gemini_recommendations"] = df["sample"].map(GEMINI_RECOMMENDATIONS)

# Remove rows where ALL systems have no recommendations
df = df[
    df["pipeline_recommendations"].apply(lambda x: isinstance(x, list) and len(x) > 0) |
    df["chatgpt_recommendations"].apply(lambda x: isinstance(x, list) and len(x) > 0) |
    df["gemini_recommendations"].apply(lambda x: isinstance(x, list) and len(x) > 0)
].copy()
# -----------------------
# 4) BUILD SUMMARY TABLE
# -----------------------
summary_rows = [
    summarize_source(df, "pipeline_recommendations", "pipeline", k=K),
    summarize_source(df, "chatgpt_recommendations", "chatgpt", k=K),
    summarize_source(df, "gemini_recommendations", "gemini", k=K),
]

summary_df = pd.DataFrame(summary_rows)

# Optional ordering
summary_df = summary_df[
    [
        "source",
        "evaluated_samples",
        "accuracy",
        f"precision@{K}",
        f"recall@{K}",
        f"ndcg@{K}",
        "map",
        f"coverage@{K}",
        "overlap_with_pipeline",
    ]
]

print("\n=== RECOMMENDATION METRICS SUMMARY ===")
print(summary_df.to_string(index=False))

summary_df.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8")
print(f"\nSaved summary to: {OUT_CSV_PATH}")