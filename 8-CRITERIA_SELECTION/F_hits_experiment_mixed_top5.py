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

K = 10
HOWMANYEXPERIMENTS = 1

CSV_PATH = "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/merged_2.csv"
FILTER_JSON_PATH = "11-RECOMMENDATION_EVALUATION/OUTPUT_F.json"

RUN_A = "experiment_runs_XX_dedup"
RUN_D = "experiment_runs_XX_dedup"

EXPERIMENT_ROOT_A = f"8-CRITERIA_SELECTION/experiments/{RUN_A}"
EXPERIMENT_ROOT_D = f"8-CRITERIA_SELECTION/experiments/{RUN_D}"

# Samples that should use eval D; all others use eval A
SPECIFIC_DENEME = {
    "9", "18", "62", "86", "90", "96",
    "116", "130", "136", "146", "177",
    "178", "179", "184", "212", "213",
    "223", "224"
}
SPECIFIC_DENEME_NORM = {f"A{int(x)}" for x in SPECIFIC_DENEME}

# Output tag for mixed-run evaluation
RUN_TAG = "experiment_runs_A_plus_D_split"

# Folder containing model metadata dictionaries
MODEL_META_DIR = "HF-Models-T7-U"

OUT_DETAIL_CSV = f"8-CRITERIA_SELECTION/hits_cluster/{RUN_TAG}/experiment_sample_stats.csv"
OUT_EXPERIMENT_CSV = f"8-CRITERIA_SELECTION/hits_cluster/{RUN_TAG}/experiment_stats_summary.csv"
OUT_SAMPLE_CSV = f"8-CRITERIA_SELECTION/hits_cluster/{RUN_TAG}/sample_stats_summary.csv"
OUT_TOPN_DETAIL_CSV = f"8-CRITERIA_SELECTION/hits_cluster/{RUN_TAG}/top_{HOWMANYEXPERIMENTS}_experiments_by_family_root_first_rank.csv"
OUT_TOP_BY_RANK_DETAIL_CSV = f"8-CRITERIA_SELECTION/hits_cluster/{RUN_TAG}/top_experiments_by_exact_rank_with_common_tiebreak.csv"

OUT_PER_SAMPLE_TOP_RANK_CSV = f"8-CRITERIA_SELECTION/hits_cluster/{RUN_TAG}/per_sample_top_{HOWMANYEXPERIMENTS}_exact_rank.csv"
OUT_PER_SAMPLE_TOP_FAMILY_CSV = f"8-CRITERIA_SELECTION/hits_cluster/{RUN_TAG}/per_sample_top_{HOWMANYEXPERIMENTS}_family_root_rank.csv"

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

    s = str(x).strip().lower()
    s = re.sub(r"\s+", "", s)

    patterns = [
        r"sample[_-]?(\d+)$",   # sample1, sample_1, sample-1
        r"[a-z](\d+)$",         # a1, d1, z12
        r"[a-z]+[_-]?(\d+)$",   # trial1, run_2, group-7
        r"(\d+)$",              # 1, 12
    ]

    for p in patterns:
        m = re.fullmatch(p, s, flags=re.IGNORECASE)
        if m:
            return f"A{int(m.group(1))}"

    return str(x).strip()


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
      run='experiment_runs_G', sample='A7', experiment='exp_001'
    """
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)

    if "experiments" not in parts:
        return None, None, None

    idx = parts.index("experiments")

    # experiments / RUN / sample / exp_xxx / response.json
    if len(parts) <= idx + 3:
        return None, None, None

    run = parts[idx + 1]
    sample = parts[idx + 2]
    experiment = parts[idx + 3]
    return run, sample, experiment


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

        if model_id in meta:
            print(f"Duplicate metadata for modelID {model_id}; overwriting with {path}")

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


def compare_gt_to_recommendations(gt_model_id, recs, model_meta, k=10):
    """
    For one GT model and its recommendation list:
      - compare family_root / assigned_modality / task
      - compute count, rate, hit@k, first matching rank, ndcg-like and ap-like scores

    Note:
      Count/rate are computed over top-k recommendations, not the full list.
    """
    gt_attrs = get_model_attrs(gt_model_id, model_meta)
    recs_k = recs[:k] if isinstance(recs, list) else []

    hit_attrs = [get_model_attrs(rec, model_meta) for rec in recs_k]

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


def sample_sort_key(col):
    return pd.to_numeric(
        col.astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    ).fillna(10**9)


def sort_samples_naturally(df_in, sample_col="sample"):
    return df_in.sort_values(by=sample_col, key=sample_sort_key)


def reorder_rank_columns(df_in):
    preferred_front = ["sample", "experiment", "rank", "family_root_first_rank"]
    existing_front = [c for c in preferred_front if c in df_in.columns]
    remaining = [c for c in df_in.columns if c not in existing_front]
    return df_in[existing_front + remaining]


def add_global_frequency_for_best(df_in, metric_col, freq_col_name):
    best_per_sample = (
        df_in.groupby("sample", dropna=False)[metric_col]
        .min()
        .reset_index()
        .rename(columns={metric_col: f"best_{metric_col}"})
    )

    out = df_in.merge(best_per_sample, on="sample", how="left")

    out["_is_best_for_sample"] = (
        out[metric_col].notna() &
        out[f"best_{metric_col}"].notna() &
        (out[metric_col] == out[f"best_{metric_col}"])
    )

    freq_df = (
        out[out["_is_best_for_sample"]]
        .groupby("experiment", dropna=False)
        .agg(**{freq_col_name: ("sample", "nunique")})
        .reset_index()
    )

    out = out.merge(freq_df, on="experiment", how="left")
    out[freq_col_name] = out[freq_col_name].fillna(0).astype(int)

    return out


def build_per_sample_top_n(df_in, metric_col, freq_col_name, rank_label_col, n):
    """
    For each sample:
      - sort experiments by:
          1) metric (lower is better)
          2) global frequency (higher is better)
          3) rank
          4) family_root_first_rank
      - take top n experiments
    """
    work = add_global_frequency_for_best(df_in.copy(), metric_col, freq_col_name)

    work = work.sort_values(
        by=[
            "sample",
            metric_col,
            freq_col_name,
            "rank",
            "family_root_first_rank",
            "experiment",
        ],
        ascending=[True, True, False, True, True, True],
        na_position="last"
    )

    out = (
        work.groupby("sample", as_index=False)
        .head(n)
        .copy()
    )

    out[rank_label_col] = (
        out.groupby("sample", sort=False).cumcount() + 1
    )

    out = reorder_rank_columns(out)

    out = out.sort_values(
        by=[
            "sample",
            rank_label_col,
            metric_col,
            "experiment",
        ],
        na_position="last"
    )

    return sort_samples_naturally(out)


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

all_csv_samples = set(df["sample_norm"].dropna().unique().tolist())
samples_using_d = sorted(all_csv_samples & SPECIFIC_DENEME_NORM, key=lambda x: int(re.search(r"(\d+)", x).group(1)))
samples_using_a = sorted(all_csv_samples - SPECIFIC_DENEME_NORM, key=lambda x: int(re.search(r"(\d+)", x).group(1)))

print("\nSamples configured to use D:", len(samples_using_d))
print(samples_using_d)
print("\nSamples configured to use A:", len(samples_using_a))
print(samples_using_a)


# -----------------------
# 2) LOAD MODEL METADATA
# -----------------------
model_meta = load_model_metadata_folder(MODEL_META_DIR)


# -----------------------
# 3) LOAD EXPERIMENT RESPONSE.JSON FILES
#    Use D for SPECIFIC_DENEME_NORM, A for all others
# -----------------------
pattern_A = os.path.join(EXPERIMENT_ROOT_A, "*", "*", "response.json")
pattern_D = os.path.join(EXPERIMENT_ROOT_D, "*", "*", "response.json")

eval_paths_A = sorted(glob.glob(pattern_A))
eval_paths_D = sorted(glob.glob(pattern_D))

print("Found A experiment response files:", len(eval_paths_A))
print("Found D experiment response files:", len(eval_paths_D))

experiment_rows = []


def process_paths(paths, expected_run_type):
    for path in paths:
        run, sample, experiment = parse_experiment_path(path)
        if run is None or sample is None or experiment is None:
            print(f"Skipping unrecognized path: {path}")
            continue

        sample_norm = normalize_sample_key(sample)

        # Keep only the paths that belong to this split
        if expected_run_type == "D":
            if sample_norm not in SPECIFIC_DENEME_NORM:
                continue
        elif expected_run_type == "A":
            if sample_norm in SPECIFIC_DENEME_NORM:
                continue

        try:
            data = load_json(path)
            recs = extract_recs_from_eval(data)
        except Exception as e:
            print(f"Could not read {path}: {e}")
            recs = []

        experiment_rows.append({
            "run_source": run,
            "run_type": expected_run_type,
            "sample": sample,
            "experiment": experiment,
            "response_path": path,
            "recommendations": recs,
            "num_recommendations": len(recs) if isinstance(recs, list) else 0,
        })


process_paths(eval_paths_D, "D")
process_paths(eval_paths_A, "A")

exp_df = pd.DataFrame(experiment_rows)

if exp_df.empty:
    raise ValueError("No experiment response.json files were found after applying the A/D sample split.")

exp_df["sample_norm"] = exp_df["sample"].apply(normalize_sample_key)

print("Loaded experiment rows after A/D split:", len(exp_df))


# -----------------------
# 4) MERGE EXPERIMENTS WITH GROUND TRUTH CSV
# -----------------------
merged = df.merge(exp_df, on="sample_norm", how="inner", suffixes=("_csv", "_exp"))
merged["sample"] = merged["sample_exp"]
merged["sample_csv"] = merged["sample_csv"]
merged["user_intent"] = merged["sample_norm"].map(intent_map)

print("\nCSV samples:")
print(sorted(df["sample_norm"].dropna().astype(str).unique().tolist(), key=lambda x: int(re.search(r"(\d+)", x).group(1))))

print("\nExperiment samples:")
print(sorted(exp_df["sample_norm"].dropna().astype(str).unique().tolist(), key=lambda x: int(re.search(r"(\d+)", x).group(1))))

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
# 7) DETAIL OUTPUT
# -----------------------
detail_cols = [
    "sample",
    "sample_norm",
    "sample_csv",
    "experiment",
    "run_type",
    "run_source",
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
# 8) SUMMARY BY EXPERIMENT
# -----------------------
experiment_summary_rows = []
for experiment, group in detail_df.groupby("experiment"):
    stats = summarize_group(group, k=K, unique_sample_col="sample_norm")
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
# 9) SUMMARY BY SAMPLE
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
].sort_values(by=["sample"], key=sample_sort_key)


# -----------------------
# 10) TOP-N EXPERIMENTS BY FAMILY_ROOT_FIRST_RANK
# -----------------------
experiment_family_root_ranking = (
    detail_df.groupby("experiment", dropna=False)
    .agg(
        family_root_first_rank_mean=("family_root_first_rank", lambda s: s.dropna().mean() if s.notna().any() else np.nan),
        family_root_first_rank_median=("family_root_first_rank", lambda s: s.dropna().median() if s.notna().any() else np.nan),
        family_root_first_rank_valid_count=("family_root_first_rank", lambda s: int(s.notna().sum())),
        family_root_hit_rate=(f"family_root_hit@{K}", "mean"),
        evaluated_rows=("experiment", "size"),
    )
    .reset_index()
)

top_n_experiments = (
    experiment_family_root_ranking.sort_values(
        by=["family_root_first_rank_mean", "family_root_first_rank_median", "family_root_hit_rate", "evaluated_rows", "experiment"],
        ascending=[True, True, False, False, True],
        na_position="last",
    )
    .head(HOWMANYEXPERIMENTS)["experiment"]
    .tolist()
)

top_n_detail_df = (
    detail_df[detail_df["experiment"].isin(top_n_experiments)]
    .copy()
)

top_n_detail_df["experiment_rank_by_family_root_first_rank"] = pd.Categorical(
    top_n_detail_df["experiment"],
    categories=top_n_experiments,
    ordered=True,
).codes + 1

top_n_detail_df = top_n_detail_df.merge(
    experiment_family_root_ranking,
    on="experiment",
    how="left",
)

top_n_detail_df = top_n_detail_df.sort_values(
    by=[
        "experiment_rank_by_family_root_first_rank",
        "sample",
        "paper",
        "title",
    ],
    key=lambda col: sample_sort_key(col) if col.name == "sample" else col
)


# -----------------------
# 11) TOP EXPERIMENTS BY EXACT RANK
#     Tie-break: experiments that are most often among the best
#     across all samples
# -----------------------
sample_best_rank = (
    detail_df.groupby("sample", dropna=False)["rank"]
    .min()
    .reset_index()
    .rename(columns={"rank": "sample_best_rank"})
)

best_rank_rows = detail_df.merge(sample_best_rank, on="sample", how="left")
best_rank_rows["is_best_for_sample"] = (
    best_rank_rows["rank"].notna() &
    best_rank_rows["sample_best_rank"].notna() &
    (best_rank_rows["rank"] == best_rank_rows["sample_best_rank"])
)

experiment_best_rank_frequency = (
    best_rank_rows[best_rank_rows["is_best_for_sample"]]
    .groupby("experiment", dropna=False)
    .agg(best_rank_sample_count=("sample", "nunique"))
    .reset_index()
)

experiment_exact_rank_ranking = (
    detail_df.groupby("experiment", dropna=False)
    .agg(
        rank_mean=("rank", lambda s: s.dropna().mean() if s.notna().any() else np.nan),
        rank_median=("rank", lambda s: s.dropna().median() if s.notna().any() else np.nan),
        rank_valid_count=("rank", lambda s: int(s.notna().sum())),
        hit_rate=(f"hit@{K}", "mean"),
        family_root_first_rank_mean=("family_root_first_rank", lambda s: s.dropna().mean() if s.notna().any() else np.nan),
        family_root_first_rank_median=("family_root_first_rank", lambda s: s.dropna().median() if s.notna().any() else np.nan),
        family_root_hit_rate=(f"family_root_hit@{K}", "mean"),
        evaluated_rows=("experiment", "size"),
    )
    .reset_index()
)

experiment_exact_rank_ranking = experiment_exact_rank_ranking.merge(
    experiment_best_rank_frequency,
    on="experiment",
    how="left",
)

experiment_exact_rank_ranking["best_rank_sample_count"] = (
    experiment_exact_rank_ranking["best_rank_sample_count"].fillna(0).astype(int)
)

top_rank_experiments = (
    experiment_exact_rank_ranking.sort_values(
        by=[
            "rank_mean",
            "rank_median",
            "best_rank_sample_count",
            "hit_rate",
            "evaluated_rows",
            "experiment",
        ],
        ascending=[True, True, False, False, False, True],
        na_position="last",
    )
    .head(HOWMANYEXPERIMENTS)["experiment"]
    .tolist()
)

top_rank_detail_df = (
    detail_df[detail_df["experiment"].isin(top_rank_experiments)]
    .copy()
)

top_rank_detail_df["experiment_rank_by_exact_rank"] = pd.Categorical(
    top_rank_detail_df["experiment"],
    categories=top_rank_experiments,
    ordered=True,
).codes + 1

top_rank_detail_df = top_rank_detail_df.merge(
    experiment_exact_rank_ranking,
    on="experiment",
    how="left",
)

front_cols = [
    "sample",
    "experiment",
    "rank",
    "family_root_first_rank",
]

remaining_cols = [c for c in top_rank_detail_df.columns if c not in front_cols]
top_rank_detail_df = top_rank_detail_df[front_cols + remaining_cols]

top_rank_detail_df = top_rank_detail_df.sort_values(
    by=[
        "experiment_rank_by_exact_rank",
        "sample",
        "paper",
        "title",
    ],
    key=lambda col: sample_sort_key(col) if col.name == "sample" else col
)


# -----------------------
# 12) PER-SAMPLE TOP-N EXPERIMENTS
# -----------------------
per_sample_top_rank_df = build_per_sample_top_n(
    detail_df,
    metric_col="rank",
    freq_col_name="best_exact_rank_sample_count",
    rank_label_col="per_sample_exact_rank_position",
    n=HOWMANYEXPERIMENTS
)

per_sample_top_family_df = build_per_sample_top_n(
    detail_df,
    metric_col="family_root_first_rank",
    freq_col_name="best_family_root_rank_sample_count",
    rank_label_col="per_sample_family_root_position",
    n=HOWMANYEXPERIMENTS
)


# -----------------------
# 13) SAVE
# -----------------------
os.makedirs(os.path.dirname(OUT_DETAIL_CSV), exist_ok=True)

detail_df = detail_df.sort_values(
    by=["sample", "experiment"],
    key=lambda col: sample_sort_key(col) if col.name == "sample" else col
)
detail_df.to_csv(OUT_DETAIL_CSV, index=False, encoding="utf-8")

experiment_summary_df.to_csv(OUT_EXPERIMENT_CSV, index=False, encoding="utf-8")

sample_summary_df = sample_summary_df.sort_values(
    by="sample",
    key=sample_sort_key
)
sample_summary_df.to_csv(OUT_SAMPLE_CSV, index=False, encoding="utf-8")

top_n_detail_df = top_n_detail_df.sort_values(
    by=["experiment_rank_by_family_root_first_rank", "sample", "paper", "title"],
    key=lambda col: sample_sort_key(col) if col.name == "sample" else col
)
top_n_detail_df.to_csv(OUT_TOPN_DETAIL_CSV, index=False, encoding="utf-8")

top_rank_detail_df = top_rank_detail_df.sort_values(
    by=["experiment_rank_by_exact_rank", "sample", "paper", "title"],
    key=lambda col: sample_sort_key(col) if col.name == "sample" else col
)
top_rank_detail_df.to_csv(OUT_TOP_BY_RANK_DETAIL_CSV, index=False, encoding="utf-8")

per_sample_top_rank_df.to_csv(
    OUT_PER_SAMPLE_TOP_RANK_CSV,
    index=False,
    encoding="utf-8"
)

per_sample_top_family_df.to_csv(
    OUT_PER_SAMPLE_TOP_FAMILY_CSV,
    index=False,
    encoding="utf-8"
)

print(f"\nSaved detail CSV to: {OUT_DETAIL_CSV}")
print(f"Saved experiment summary CSV to: {OUT_EXPERIMENT_CSV}")
print(f"Saved sample summary CSV to: {OUT_SAMPLE_CSV}")
print(f"Saved top-{HOWMANYEXPERIMENTS}-by-family-root detail CSV to: {OUT_TOPN_DETAIL_CSV}")
print(f"Saved top-by-rank detail CSV to: {OUT_TOP_BY_RANK_DETAIL_CSV}")
print(f"Saved per-sample TOP-{HOWMANYEXPERIMENTS} exact-rank CSV to: {OUT_PER_SAMPLE_TOP_RANK_CSV}")
print(f"Saved per-sample TOP-{HOWMANYEXPERIMENTS} family-root CSV to: {OUT_PER_SAMPLE_TOP_FAMILY_CSV}")