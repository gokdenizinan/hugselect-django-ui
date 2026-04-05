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

RUN = "experiment_runs_D99"
EXPERIMENT_ROOT = f"8-CRITERIA_SELECTION/experiments/{RUN}"

# Folder containing model metadata dictionaries
MODEL_META_DIR = "HF-Models-T7-U"

OUT_DETAIL_CSV = f"8-CRITERIA_SELECTION/hits_cluster/{RUN}/experiment_sample_stats.csv"
OUT_EXPERIMENT_CSV = f"8-CRITERIA_SELECTION/hits_cluster/{RUN}/experiment_stats_summary.csv"
OUT_SAMPLE_CSV = f"8-CRITERIA_SELECTION/hits_cluster/{RUN}/sample_stats_summary.csv"

K = 10
KEEP_COLS = ["sample", "paper", "title", "modelID", "year", "venue"]
ATTR_KEYS = ["family_root", "assigned_modality", "task"]


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


def normalize_attr_value(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None

    if isinstance(x, list):
        vals = []
        for v in x:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            vals.append(str(v).strip().lower())
        return vals if vals else None

    return str(x).strip().lower()


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
      .../8-CRITERIA_SELECTION/experiments/experiment_runs_G/A7/exp_001/response.json
    Returns:
      sample='A7', experiment='exp_001'
    """
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)

    if RUN not in parts:
        return None, None

    idx = parts.index(RUN)
    if len(parts) <= idx + 2:
        return None, None

    sample = parts[idx + 1]
    experiment = parts[idx + 2]
    return sample, experiment


def find_first_dict_with_keys(obj, required_keys):
    """
    Recursively find the first dict that contains all required keys.
    Useful when metadata files have slightly nested structures.
    """
    if isinstance(obj, dict):
        if all(k in obj for k in required_keys):
            return obj

        for v in obj.values():
            found = find_first_dict_with_keys(v, required_keys)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = find_first_dict_with_keys(item, required_keys)
            if found is not None:
                return found

    return None


def find_clusters_dict(obj):
    """
    Recursively find a dict under key 'Clusters' or 'clusters'.
    """
    if isinstance(obj, dict):
        if "Clusters" in obj and isinstance(obj["Clusters"], dict):
            return obj["Clusters"]
        if "clusters" in obj and isinstance(obj["clusters"], dict):
            return obj["clusters"]

        for v in obj.values():
            found = find_clusters_dict(v)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = find_clusters_dict(item)
            if found is not None:
                return found

    return None


def load_model_metadata_folder(folder_path):
    """
    Loads all JSON files in a folder recursively and returns:
      {
        normalized_model_id: {
          "family_root": ...,
          "assigned_modality": ...,
          "task": ...
        }
      }

    Expected somewhere in each file:
      - key modelID
      - key Clusters
      - inside Clusters: family_root, assigned_modality, task
    """
    meta = {}
    json_paths = glob.glob(os.path.join(folder_path, "**", "*.json"), recursive=True)

    print("Found metadata files:", len(json_paths))

    for path in json_paths:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"Could not read metadata file {path}: {e}")
            continue

        model_dict = find_first_dict_with_keys(data, ["modelID"])
        if model_dict is None:
            print(f"Skipping metadata file with no modelID: {path}")
            continue

        model_id = normalize_model_string(model_dict.get("modelID"))
        if not model_id:
            print(f"Skipping metadata file with empty modelID: {path}")
            continue

        clusters = find_clusters_dict(data)
        if clusters is None:
            clusters = {}

        meta[model_id] = {
            "family_root": clusters.get("family_root"),
            "assigned_modality": clusters.get("assigned_modality"),
            "task": clusters.get("task"),
        }

    print("Loaded model metadata entries:", len(meta))
    return meta


def get_model_attrs(model_id, model_meta):
    model_id = normalize_model_string(model_id)
    if model_id is None:
        return {k: None for k in ATTR_KEYS}
    return model_meta.get(model_id, {k: None for k in ATTR_KEYS})


def attr_match(gt_val, hit_val):
    """
    Supports:
      scalar vs scalar
      scalar vs list
      list vs scalar
      list vs list
    """
    gt_val = normalize_attr_value(gt_val)
    hit_val = normalize_attr_value(hit_val)

    if gt_val is None or hit_val is None:
        return np.nan

    if isinstance(gt_val, list) and isinstance(hit_val, list):
        return 1.0 if len(set(gt_val) & set(hit_val)) > 0 else 0.0

    if isinstance(gt_val, list):
        return 1.0 if hit_val in gt_val else 0.0

    if isinstance(hit_val, list):
        return 1.0 if gt_val in hit_val else 0.0

    return 1.0 if gt_val == hit_val else 0.0


def first_matching_rank_for_attr(gt_model_id, recs, model_meta, attr_key):
    """
    Returns the first rank where the recommended model matches the GT model
    on the given attribute. Rank is 1-based. Returns NaN if no match.
    """
    gt_attrs = get_model_attrs(gt_model_id, model_meta)
    gt_val = gt_attrs.get(attr_key)

    if not isinstance(recs, list):
        return np.nan

    for idx, rec_model in enumerate(recs, start=1):
        rec_attrs = get_model_attrs(rec_model, model_meta)
        rec_val = rec_attrs.get(attr_key)
        m = attr_match(gt_val, rec_val)
        if m == 1.0:
            return idx

    return np.nan


def compare_gt_to_recommendations(gt_model_id, recs, model_meta, k=10):
    """
    For one GT model and its recommendation list:
      - compare family_root / assigned_modality / task
      - compute count, rate, hit@k, first matching rank, ndcg-like and ap-like scores
    """
    gt_attrs = get_model_attrs(gt_model_id, model_meta)

    hit_attrs = []
    for rec in recs if isinstance(recs, list) else []:
        hit_attrs.append(get_model_attrs(rec, model_meta))

    out = {}

    for key in ATTR_KEYS:
        gt_val = gt_attrs.get(key)
        matches = [attr_match(gt_val, h.get(key)) for h in hit_attrs]
        valid_matches = [m for m in matches if not pd.isna(m)]

        first_rank = np.nan
        for idx, m in enumerate(matches, start=1):
            if m == 1.0:
                first_rank = idx
                break

        out[f"gt_{key}"] = gt_val
        out[f"{key}_matches_count"] = int(sum(valid_matches)) if valid_matches else 0
        out[f"{key}_match_rate"] = round(float(np.mean(valid_matches)), 4) if valid_matches else np.nan
        out[f"{key}_first_rank"] = first_rank
        out[f"{key}_hit@{k}"] = 0.0 if pd.isna(first_rank) or first_rank > k else 1.0
        out[f"{key}_precision@{k}"] = 0.0 if pd.isna(first_rank) or first_rank > k else 1.0 / k
        out[f"{key}_recall@{k}"] = 0.0 if pd.isna(first_rank) or first_rank > k else 1.0
        out[f"{key}_ndcg@{k}"] = 0.0 if pd.isna(first_rank) or first_rank > k else 1.0 / math.log2(first_rank + 1)
        out[f"{key}_map@{k}"] = 0.0 if pd.isna(first_rank) or first_rank > k else 1.0 / first_rank

    return out


def summarize_group(df_group, k=10, unique_sample_col=None):
    if len(df_group) == 0:
        base = {
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

        for key in ATTR_KEYS:
            base.update({
                f"{key}_match_rate": np.nan,
                f"{key}_mean_first_rank": np.nan,
                f"{key}_median_first_rank": np.nan,
                f"{key}_precision@{k}": 0.0,
                f"{key}_recall@{k}": 0.0,
                f"{key}_ndcg@{k}": 0.0,
                f"{key}_map@{k}": 0.0,
                f"{key}_hit@{k}": 0.0,
            })
        return base

    evaluated_samples = (
        int(df_group[unique_sample_col].nunique())
        if unique_sample_col and unique_sample_col in df_group.columns
        else int(len(df_group))
    )

    out = {
        "evaluated_samples": evaluated_samples,
        "accuracy@1": round(df_group["accuracy@1"].mean(), 4),
        f"precision@{k}": round(df_group[f"precision@{k}"].mean(), 4),
        f"recall@{k}": round(df_group[f"recall@{k}"].mean(), 4),
        f"ndcg@{k}": round(df_group[f"ndcg@{k}"].mean(), 4),
        f"map@{k}": round(df_group[f"map@{k}"].mean(), 4),
        f"hit@{k}": round(df_group[f"hit@{k}"].mean(), 4),
        "mean_rank": round(df_group["rank"].dropna().mean(), 4) if df_group["rank"].notna().any() else np.nan,
        "median_rank": round(df_group["rank"].dropna().median(), 4) if df_group["rank"].notna().any() else np.nan,
    }

    for key in ATTR_KEYS:
        out[f"{key}_match_rate"] = (
            round(df_group[f"{key}_match_rate"].dropna().mean(), 4)
            if df_group[f"{key}_match_rate"].notna().any() else np.nan
        )
        out[f"{key}_mean_first_rank"] = (
            round(df_group[f"{key}_first_rank"].dropna().mean(), 4)
            if df_group[f"{key}_first_rank"].notna().any() else np.nan
        )
        out[f"{key}_median_first_rank"] = (
            round(df_group[f"{key}_first_rank"].dropna().median(), 4)
            if df_group[f"{key}_first_rank"].notna().any() else np.nan
        )
        out[f"{key}_precision@{k}"] = round(df_group[f"{key}_precision@{k}"].mean(), 4)
        out[f"{key}_recall@{k}"] = round(df_group[f"{key}_recall@{k}"].mean(), 4)
        out[f"{key}_ndcg@{k}"] = round(df_group[f"{key}_ndcg@{k}"].mean(), 4)
        out[f"{key}_map@{k}"] = round(df_group[f"{key}_map@{k}"].mean(), 4)
        out[f"{key}_hit@{k}"] = round(df_group[f"{key}_hit@{k}"].mean(), 4)

    return out


# -----------------------
# 1) LOAD BASE CSV
# -----------------------
df = pd.read_csv(CSV_PATH, sep=";", engine="python")

df = df.rename(columns={
    "saved_name": "paper",
    "matched_hf_model": "modelID"
})

missing = [c for c in KEEP_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Input CSV is missing required columns: {missing}")

df = df[KEEP_COLS].copy()
df["sample_norm"] = df["sample"].apply(normalize_sample_key)
df["modelID"] = df["modelID"].apply(normalize_model_string)

intent_map = {}

if os.path.exists(FILTER_JSON_PATH):
    filter_data = load_json(FILTER_JSON_PATH)
    intent_map = {
        normalize_sample_key(k): v.get("user_intent")
        for k, v in filter_data.items()
    }
    valid_samples = {normalize_sample_key(k) for k in filter_data.keys()}

    print("Samples allowed by filter:", len(valid_samples))
    df = df[df["sample_norm"].isin(valid_samples)].copy()

print("Rows after CSV/filter load:", len(df))


# -----------------------
# 2) LOAD MODEL METADATA
# -----------------------
model_meta = load_model_metadata_folder(MODEL_META_DIR)


# -----------------------
# 3) LOAD ALL EXPERIMENT RESPONSE.JSON FILES
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

if exp_df.empty:
    raise ValueError("No experiment response.json files were found.")

exp_df["sample_norm"] = exp_df["sample"].apply(normalize_sample_key)

print("Loaded experiment rows:", len(exp_df))


# -----------------------
# 4) MERGE EXPERIMENTS WITH GROUND TRUTH CSV
# -----------------------
merged = df.merge(exp_df, on="sample_norm", how="inner", suffixes=("_csv", "_exp"))
merged["sample"] = merged["sample_exp"]
merged["user_intent"] = merged["sample_norm"].map(intent_map)

print("\nCSV samples:")
print(sorted(df["sample_norm"].dropna().astype(str).unique().tolist()))

print("\nExperiment samples:")
print(sorted(exp_df["sample_norm"].dropna().astype(str).unique().tolist()))

merged = merged[
    merged["recommendations"].apply(lambda x: isinstance(x, list) and len(x) > 0)
].copy()

print("Rows after merging experiments with samples:", len(merged))

if merged.empty:
    raise ValueError("Merged dataframe is empty. Check sample names in CSV and folder structure.")


# -----------------------
# 5) COMPUTE EXACT MODEL METRICS
# -----------------------
merged["rank"] = merged.apply(
    lambda row: find_rank(row["modelID"], row["recommendations"]),
    axis=1
)

merged["accuracy@1"] = merged["rank"].apply(accuracy_at_1)
merged[f"precision@{K}"] = merged["rank"].apply(lambda r: precision_at_k(r, K))
merged[f"recall@{K}"] = merged["rank"].apply(lambda r: recall_at_k(r, K))
merged[f"ndcg@{K}"] = merged["rank"].apply(lambda r: ndcg_at_k(r, K))
merged[f"map@{K}"] = merged["rank"].apply(lambda r: average_precision_at_k(r, K))
merged[f"hit@{K}"] = merged["rank"].apply(lambda r: hit_at_k(r, K))


# -----------------------
# 6) COMPUTE ATTRIBUTE-LEVEL MATCH METRICS
# -----------------------
attr_comparisons = merged.apply(
    lambda row: compare_gt_to_recommendations(
        row["modelID"],
        row["recommendations"],
        model_meta,
        k=K
    ),
    axis=1
)

attr_df = pd.DataFrame(attr_comparisons.tolist(), index=merged.index)
merged = pd.concat([merged, attr_df], axis=1)


# -----------------------
# 7) OPTIONAL DEBUG OUTPUT
# -----------------------
# missing_exact = merged[merged["rank"].isna()][
#     ["sample", "experiment", "paper", "title", "modelID", "recommendations"]
# ].copy()

# print("\n=== ROWS WITH NO EXACT MODEL HIT ===")
# if len(missing_exact) == 0:
#     print("None")
# else:
#     print(missing_exact.head(20).to_string(index=False))

# for key in ATTR_KEYS:
#     col = f"{key}_first_rank"
#     no_attr_hit = merged[merged[col].isna()][["sample", "experiment", "modelID", "recommendations"]].copy()

#     print(f"\n=== ROWS WITH NO {key.upper()} HIT ===")
#     if len(no_attr_hit) == 0:
#         print("None")
#     else:
#         print(no_attr_hit.head(20).to_string(index=False))


# -----------------------
# 8) DETAIL OUTPUT
# -----------------------
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

    "gt_family_root",
    "family_root_matches_count",
    "family_root_match_rate",
    "family_root_first_rank",
    f"family_root_precision@{K}",
    f"family_root_recall@{K}",
    f"family_root_ndcg@{K}",
    f"family_root_map@{K}",
    f"family_root_hit@{K}",

    "gt_assigned_modality",
    "assigned_modality_matches_count",
    "assigned_modality_match_rate",
    "assigned_modality_first_rank",
    f"assigned_modality_precision@{K}",
    f"assigned_modality_recall@{K}",
    f"assigned_modality_ndcg@{K}",
    f"assigned_modality_map@{K}",
    f"assigned_modality_hit@{K}",

    "gt_task",
    "task_matches_count",
    "task_match_rate",
    "task_first_rank",
    f"task_precision@{K}",
    f"task_recall@{K}",
    f"task_ndcg@{K}",
    f"task_map@{K}",
    f"task_hit@{K}",

    "response_path",
]
detail_df = merged[detail_cols].copy()


# -----------------------
# 9) SUMMARY BY EXPERIMENT
# -----------------------
experiment_summary_rows = []
for experiment, group in detail_df.groupby("experiment"):
    stats = summarize_group(group, k=K, unique_sample_col="sample")
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

        "family_root_match_rate",
        "family_root_mean_first_rank",
        "family_root_median_first_rank",
        f"family_root_precision@{K}",
        f"family_root_recall@{K}",
        f"family_root_ndcg@{K}",
        f"family_root_map@{K}",
        f"family_root_hit@{K}",

        "assigned_modality_match_rate",
        "assigned_modality_mean_first_rank",
        "assigned_modality_median_first_rank",
        f"assigned_modality_precision@{K}",
        f"assigned_modality_recall@{K}",
        f"assigned_modality_ndcg@{K}",
        f"assigned_modality_map@{K}",
        f"assigned_modality_hit@{K}",

        "task_match_rate",
        "task_mean_first_rank",
        "task_median_first_rank",
        f"task_precision@{K}",
        f"task_recall@{K}",
        f"task_ndcg@{K}",
        f"task_map@{K}",
        f"task_hit@{K}",
    ]
].sort_values(by=["experiment"])


# -----------------------
# 10) SUMMARY BY SAMPLE
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

        "family_root_match_rate",
        "family_root_mean_first_rank",
        "family_root_median_first_rank",
        f"family_root_precision@{K}",
        f"family_root_recall@{K}",
        f"family_root_ndcg@{K}",
        f"family_root_map@{K}",
        f"family_root_hit@{K}",

        "assigned_modality_match_rate",
        "assigned_modality_mean_first_rank",
        "assigned_modality_median_first_rank",
        f"assigned_modality_precision@{K}",
        f"assigned_modality_recall@{K}",
        f"assigned_modality_ndcg@{K}",
        f"assigned_modality_map@{K}",
        f"assigned_modality_hit@{K}",

        "task_match_rate",
        "task_mean_first_rank",
        "task_median_first_rank",
        f"task_precision@{K}",
        f"task_recall@{K}",
        f"task_ndcg@{K}",
        f"task_map@{K}",
        f"task_hit@{K}",
    ]
].sort_values(by=["sample"])


# -----------------------
# 11) SAVE
# -----------------------
os.makedirs(os.path.dirname(OUT_DETAIL_CSV), exist_ok=True)

detail_df.to_csv(OUT_DETAIL_CSV, index=False, encoding="utf-8")
experiment_summary_df.to_csv(OUT_EXPERIMENT_CSV, index=False, encoding="utf-8")
sample_summary_df.to_csv(OUT_SAMPLE_CSV, index=False, encoding="utf-8")

# print("\n=== DETAIL (sample + experiment) ===")
# print(detail_df.head(20).to_string(index=False))

# print("\n=== SUMMARY BY EXPERIMENT ===")
# print(experiment_summary_df.to_string(index=False))

# print("\n=== SUMMARY BY SAMPLE ===")
# print(sample_summary_df.to_string(index=False))

print(f"\nSaved detail CSV to: {OUT_DETAIL_CSV}")
print(f"Saved experiment summary CSV to: {OUT_EXPERIMENT_CSV}")
print(f"Saved sample summary CSV to: {OUT_SAMPLE_CSV}")