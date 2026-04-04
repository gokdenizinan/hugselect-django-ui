import json
import glob
import os
import math
import pandas as pd
import numpy as np
import re
# -----------------------
# CONFIG
# -----------------------
CSV_PATH = "11-RECOMMENDATION_EVALUATION/paper_model_2/snipped_papers_6.csv"
FILTER_JSON_PATH = "11-RECOMMENDATION_EVALUATION/OUTPUT_F.json"

RUN = "experiment_runs_G"
EXPERIMENT_ROOT = f"8-CRITERIA_SELECTION/experiments/{RUN}"

OUT_DETAIL_CSV = f"8-CRITERIA_SELECTION/hits/{RUN}/experiment_sample_stats.csv"
OUT_EXPERIMENT_CSV = f"8-CRITERIA_SELECTION/hits/{RUN}/experiment_stats_summary.csv"
OUT_SAMPLE_CSV = f"8-CRITERIA_SELECTION/hits/{RUN}/sample_stats_summary.csv"

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

def normalize_sample_key(x):
    if pd.isna(x):
        return None

    s = str(x).strip()

    m = re.fullmatch(r"sample[_-]?(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"A{int(m.group(1))}"

    m = re.fullmatch(r"A(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"A{int(m.group(1))}"

    return s


def extract_recs_from_eval(eval_data: dict):
    hits = eval_data.get("hits", {}).get("hits", [])
    pretty_ids = []
    for item in hits:
        pretty_id = item.get("pretty_id")
        if pretty_id is not None:
            pretty_ids.append(pretty_id)
    return pretty_ids


def find_rank(target_model, recs):
    target = normalize_model_string(target_model)
    if target is None or not isinstance(recs, list):
        return np.nan

    recs_norm = [normalize_model_string(x) for x in recs]
    try:
        return recs_norm.index(target) + 1  # 1-based
    except ValueError:
        return np.nan


def accuracy_at_1(rank):
    return 1.0 if rank == 1 else 0.0


def precision_at_k(rank, k=10):
    if pd.isna(rank) or rank > k:
        return 0.0
    return 1.0 / k


def recall_at_k(rank, k=10):
    if pd.isna(rank) or rank > k:
        return 0.0
    return 1.0


def ndcg_at_k(rank, k=10):
    if pd.isna(rank) or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def average_precision_at_k(rank, k=10):
    if pd.isna(rank) or rank > k:
        return 0.0
    return 1.0 / rank


def hit_at_k(rank, k=10):
    return 0.0 if pd.isna(rank) or rank > k else 1.0


def parse_experiment_path(path):
    """
    Expected:
    .../experiment_runs/A7/exp_001/response.json
    Returns:
      sample='A7', experiment='exp_001'
    """
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)

    # find experiment_runs index safely
    if RUN not in parts:
        return None, None

    idx = parts.index(RUN)
    if len(parts) <= idx + 2:
        return None, None

    sample = parts[idx + 1]
    experiment = parts[idx + 2]
    return sample, experiment


def summarize_group(df_group, k=10):
    if len(df_group) == 0:
        return {
            "evaluated_samples": 0,
            "accuracy@1": 0.0,
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            f"map@{k}": 0.0,
            f"hit@{k}": 0.0,
            "mean_rank": np.nan,
            "median_rank": np.nan,
        }

    return {
        "evaluated_samples": int(len(df_group)),
        "accuracy@1": round(df_group["accuracy@1"].mean(), 4),
        f"precision@{k}": round(df_group[f"precision@{k}"].mean(), 4),
        f"recall@{k}": round(df_group[f"recall@{k}"].mean(), 4),
        f"ndcg@{k}": round(df_group[f"ndcg@{k}"].mean(), 4),
        f"map@{k}": round(df_group[f"map@{k}"].mean(), 4),
        f"hit@{k}": round(df_group[f"hit@{k}"].mean(), 4),
        "mean_rank": round(df_group["rank"].dropna().mean(), 4) if df_group["rank"].notna().any() else np.nan,
        "median_rank": round(df_group["rank"].dropna().median(), 4) if df_group["rank"].notna().any() else np.nan,
    }


# -----------------------
# 1) LOAD BASE CSV
# -----------------------
df = pd.read_csv(CSV_PATH, sep=";", engine="python")

df["sample_norm"] = df["sample"].apply(normalize_sample_key)

df = df.rename(columns={
    "saved_name": "paper",
    "matched_hf_model": "modelID"
})

missing = [c for c in KEEP_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Input CSV is missing required columns: {missing}")

df = df[KEEP_COLS].copy()
df["modelID"] = df["modelID"].apply(normalize_model_string)

# optional filter from OUTPUT_F.json
if os.path.exists(FILTER_JSON_PATH):
    filter_data = load_json(FILTER_JSON_PATH)
    intent_map = {
    normalize_sample_key(k): v.get("user_intent")
    for k, v in filter_data.items()
}
    valid_samples = {normalize_sample_key(k) for k in filter_data.keys()}
    print("Samples allowed by filter:", len(valid_samples))

    df["sample_norm"] = df["sample"].apply(normalize_sample_key)
    df = df[df["sample_norm"].isin(valid_samples)].copy()

print("Rows after CSV/filter load:", len(df))


# -----------------------
# 2) LOAD ALL EXPERIMENT RESPONSE.JSON FILES
# -----------------------
pattern = os.path.join(EXPERIMENT_ROOT, "*", "*", "response.json")
eval_paths = sorted(glob.glob(pattern))

print("Found experiment response files:", len(eval_paths))

experiment_rows = []

for path in eval_paths:
    sample, experiment = parse_experiment_path(path)
    if sample is None or experiment is None:
        print(f"Skipping unrecognized path: {path}")
        continue

    try:
        data = load_json(path)
        recs = extract_recs_from_eval(data)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        recs = []

    experiment_rows.append({
        "sample": sample,
        "experiment": experiment,
        "response_path": path,
        "recommendations": recs,
        "num_recommendations": len(recs) if isinstance(recs, list) else 0,
    })

exp_df = pd.DataFrame(experiment_rows)

exp_df["sample_norm"] = exp_df["sample"].apply(normalize_sample_key)

if exp_df.empty:
    raise ValueError("No experiment response.json files were found.")

print("Loaded experiment rows:", len(exp_df))


# -----------------------
# 3) MERGE EXPERIMENTS WITH GROUND TRUTH CSV
# -----------------------
# merged = df.merge(exp_df, on="sample", how="inner")
merged = df.merge(exp_df, on="sample_norm", how="inner", suffixes=("_csv", "_exp"))
merged["sample"] = merged["sample_exp"]
merged["user_intent"] = merged["sample_norm"].map(intent_map)

print("\nCSV samples:")
print(sorted(df["sample"].astype(str).unique().tolist()))

print("\nExperiment samples:")
print(sorted(exp_df["sample"].astype(str).unique().tolist()))

# remove rows with no recommendations
merged = merged[
    merged["recommendations"].apply(lambda x: isinstance(x, list) and len(x) > 0)
].copy()

print("Rows after merging experiments with samples:", len(merged))

if merged.empty:
    raise ValueError("Merged dataframe is empty. Check sample names in CSV and folder structure.")


# -----------------------
# 4) COMPUTE PER-ROW METRICS
# -----------------------
merged["rank"] = merged.apply(lambda row: find_rank(row["modelID"], row["recommendations"]), axis=1)

merged["accuracy@1"] = merged["rank"].apply(accuracy_at_1)
merged[f"precision@{K}"] = merged["rank"].apply(lambda r: precision_at_k(r, K))
merged[f"recall@{K}"] = merged["rank"].apply(lambda r: recall_at_k(r, K))
merged[f"ndcg@{K}"] = merged["rank"].apply(lambda r: ndcg_at_k(r, K))
merged[f"map@{K}"] = merged["rank"].apply(lambda r: average_precision_at_k(r, K))
merged[f"hit@{K}"] = merged["rank"].apply(lambda r: hit_at_k(r, K))

# reorder detail columns
detail_cols = [
    "sample",
    "experiment",
    "paper",
    "title",
    "modelID",
    "user_intent",   
    "year",
    "venue",
    "num_recommendations",
    "rank",
    "accuracy@1",
    f"precision@{K}",
    f"recall@{K}",
    f"ndcg@{K}",
    f"map@{K}",
    f"hit@{K}",
    "response_path",
]
detail_df = merged[detail_cols].copy()


# -----------------------
# 5) SUMMARY BY EXPERIMENT
# -----------------------
experiment_summary_rows = []
for experiment, group in detail_df.groupby("experiment"):
    stats = summarize_group(group, k=K)
    stats["experiment"] = experiment
    experiment_summary_rows.append(stats)

experiment_summary_df = pd.DataFrame(experiment_summary_rows)
experiment_summary_df = experiment_summary_df[
    [
        "experiment",
        "evaluated_samples",
        "accuracy@1",
        f"precision@{K}",
        f"recall@{K}",
        f"ndcg@{K}",
        f"map@{K}",
        f"hit@{K}",
        "mean_rank",
        "median_rank",
    ]
].sort_values(by=["experiment"])


# -----------------------
# 6) SUMMARY BY SAMPLE
# -----------------------
sample_summary_rows = []
for sample, group in detail_df.groupby("sample"):
    stats = summarize_group(group, k=K)
    stats["sample"] = sample
    stats["num_experiments"] = group["experiment"].nunique()
    sample_summary_rows.append(stats)

sample_summary_df = pd.DataFrame(sample_summary_rows)
sample_summary_df = sample_summary_df[
    [
        "sample",
        "num_experiments",
        "evaluated_samples",
        "accuracy@1",
        f"precision@{K}",
        f"recall@{K}",
        f"ndcg@{K}",
        f"map@{K}",
        f"hit@{K}",
        "mean_rank",
        "median_rank",
    ]
].sort_values(by=["sample"])

#19 ve 56 haric 7 ile 57 arasi hit bulmali
# -----------------------
# 7) SAVE
# -----------------------
os.makedirs(os.path.dirname(OUT_DETAIL_CSV), exist_ok=True)

detail_df.to_csv(OUT_DETAIL_CSV, index=False, encoding="utf-8")
experiment_summary_df.to_csv(OUT_EXPERIMENT_CSV, index=False, encoding="utf-8")
sample_summary_df.to_csv(OUT_SAMPLE_CSV, index=False, encoding="utf-8")

print("\n=== DETAIL (sample + experiment) ===")
print(detail_df.head(20).to_string(index=False))

print("\n=== SUMMARY BY EXPERIMENT ===")
print(experiment_summary_df.to_string(index=False))

print("\n=== SUMMARY BY SAMPLE ===")
print(sample_summary_df.to_string(index=False))

print(f"\nSaved detail CSV to: {OUT_DETAIL_CSV}")
print(f"Saved experiment summary CSV to: {OUT_EXPERIMENT_CSV}")
print(f"Saved sample summary CSV to: {OUT_SAMPLE_CSV}")