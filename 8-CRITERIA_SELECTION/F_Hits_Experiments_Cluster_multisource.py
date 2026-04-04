import json
import glob
import os
import math
import pandas as pd
import numpy as np
import re
import importlib.util
from typing import Dict, List, Any, Optional

# -----------------------
# CONFIG
# -----------------------
CSV_PATH = "11-RECOMMENDATION_EVALUATION/paper_model_2/snipped_papers_6.csv"
FILTER_JSON_PATH = "11-RECOMMENDATION_EVALUATION/OUTPUT_F.json"

RUN = "experiment_runs_G"
EXPERIMENT_ROOT = f"8-CRITERIA_SELECTION/experiments/{RUN}"

# Python file containing externally collected recommendation dicts.
# Expected globals:
#   CHATGPT_RECOMMENDATIONS = {...}
#   GEMINI_RECOMMENDATIONS = {...}
EXTERNAL_RECOMMENDATIONS_PY = "8-CRITERIA_SELECTION/F_Hits_United.py"

# Folder containing model metadata dictionaries
MODEL_META_DIR = "HF-Models-T7-U"

OUT_DIR = f"8-CRITERIA_SELECTION/hits_cluster_multisource/{RUN}"
OUT_DETAIL_CSV = os.path.join(OUT_DIR, "experiment_sample_stats.csv")
OUT_EXPERIMENT_CSV = os.path.join(OUT_DIR, "experiment_stats_summary.csv")
OUT_SAMPLE_CSV = os.path.join(OUT_DIR, "sample_stats_summary.csv")
OUT_COMPACT_DETAIL_CSV = os.path.join(OUT_DIR, "experiment_sample_stats_compact.csv")

K = 10
KEEP_COLS = ["sample", "paper", "title", "modelID", "year", "venue"]
ATTR_KEYS = ["family_root", "assigned_modality", "task"]
NO_INFO = "NO INFO"

# Detail output mode:
#   "full"    -> write full detail csv only
#   "compact" -> write compact detail csv only
#   "both"    -> write both full and compact csv files
DETAIL_OUTPUT_MODE = "both"

# Which sources to evaluate.
# - "recsys" reads response.json files from EXPERIMENT_ROOT
# - "chatgpt" and "gemini" read dicts from EXTERNAL_RECOMMENDATIONS_PY
ENABLE_SOURCES = {
    "recsys": True,
    "chatgpt": True,
    "gemini": True,
}


# -----------------------
# HELPERS
# -----------------------
def load_json(path: str):
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
    if isinstance(x, str) and x.strip().upper() == NO_INFO:
        return None

    if isinstance(x, list):
        vals = []
        for v in x:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            if isinstance(v, str) and v.strip().upper() == NO_INFO:
                continue
            vals.append(str(v).strip().lower())
        return vals if vals else None

    return str(x).strip().lower()


def value_or_no_info(x):
    if x is None:
        return NO_INFO
    if isinstance(x, float) and np.isnan(x):
        return NO_INFO
    if isinstance(x, list):
        cleaned = []
        for v in x:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            cleaned.append(v)
        return cleaned if cleaned else NO_INFO
    return x


def extract_recs_from_eval(eval_data: dict):
    hits = eval_data.get("hits", {}).get("hits", [])
    pretty_ids = []
    for item in hits:
        pretty_id = item.get("pretty_id")
        if pretty_id is not None:
            pretty_ids.append(pretty_id)
    return pretty_ids


def normalize_recommendation_list(recs, k=None):
    if not isinstance(recs, list):
        return []

    out = []
    for rec in recs:
        rec_norm = normalize_model_string(rec)
        if rec_norm is None:
            continue
        out.append(rec_norm)

    if k is not None:
        return out[:k]
    return out


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
            "family_root": value_or_no_info(clusters.get("family_root")),
            "assigned_modality": value_or_no_info(clusters.get("assigned_modality")),
            "task": value_or_no_info(clusters.get("task")),
        }

    print("Loaded model metadata entries:", len(meta))
    return meta


def get_model_attrs(model_id, model_meta):
    model_id = normalize_model_string(model_id)
    if model_id is None:
        return {k: NO_INFO for k in ATTR_KEYS}
    return model_meta.get(model_id, {k: NO_INFO for k in ATTR_KEYS})


def attr_match(gt_val, hit_val):
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

        out[f"gt_{key}"] = value_or_no_info(gt_val)
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


def import_recommendation_module(py_path: str):
    if not os.path.exists(py_path):
        raise FileNotFoundError(
            f"External recommendations file not found: {py_path}. "
            "Set EXTERNAL_RECOMMENDATIONS_PY to the correct path."
        )

    module_name = os.path.splitext(os.path.basename(py_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import external recommendations module from: {py_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_external_source_rows(source_name: str, source_dict: Dict[str, List[str]], eligible_samples: set):
    rows = []
    for raw_sample, recs in source_dict.items():
        sample_norm = normalize_sample_key(raw_sample)
        if sample_norm not in eligible_samples:
            continue

        rows.append({
            "source": source_name,
            "sample": sample_norm,
            "experiment": source_name,
            "response_path": f"{EXTERNAL_RECOMMENDATIONS_PY}:{source_name.upper()}_RECOMMENDATIONS[{raw_sample}]",
            "recommendations": normalize_recommendation_list(recs, k=K),
            "num_recommendations": len(normalize_recommendation_list(recs, k=K)),
            "sample_norm": sample_norm,
        })
    return rows


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
# 3) LOAD RECSYS EXPERIMENT RESPONSE.JSON FILES
# -----------------------
pattern = os.path.join(EXPERIMENT_ROOT, "*", "*", "response.json")
eval_paths = sorted(glob.glob(pattern))

print("Found recsys experiment response files:", len(eval_paths))

experiment_rows = []

for path in eval_paths:
    sample, experiment = parse_experiment_path(path)
    if sample is None or experiment is None:
        print(f"Skipping unrecognized path: {path}")
        continue

    try:
        data = load_json(path)
        recs = normalize_recommendation_list(extract_recs_from_eval(data), k=K)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        recs = []

    experiment_rows.append({
        "source": "recsys",
        "sample": sample,
        "experiment": experiment,
        "response_path": path,
        "recommendations": recs,
        "num_recommendations": len(recs),
    })

exp_df = pd.DataFrame(experiment_rows)

if exp_df.empty:
    raise ValueError("No recsys experiment response.json files were found.")

exp_df["sample_norm"] = exp_df["sample"].apply(normalize_sample_key)
print("Loaded recsys experiment rows:", len(exp_df))

eligible_samples = set(exp_df["sample_norm"].dropna().tolist())
print("Eligible samples for external source evaluation:", len(eligible_samples))


# -----------------------
# 4) LOAD CHATGPT / GEMINI RECOMMENDATIONS
# -----------------------
all_results = []

if ENABLE_SOURCES.get("recsys", False):
    all_results.append(exp_df.copy())

if ENABLE_SOURCES.get("chatgpt", False) or ENABLE_SOURCES.get("gemini", False):
    ext_module = import_recommendation_module(EXTERNAL_RECOMMENDATIONS_PY)

    if ENABLE_SOURCES.get("chatgpt", False):
        chatgpt_dict = getattr(ext_module, "CHATGPT_RECOMMENDATIONS", {})
        chatgpt_rows = build_external_source_rows("chatgpt", chatgpt_dict, eligible_samples)
        print("Loaded chatgpt rows:", len(chatgpt_rows))
        all_results.append(pd.DataFrame(chatgpt_rows))

    if ENABLE_SOURCES.get("gemini", False):
        gemini_dict = getattr(ext_module, "GEMINI_RECOMMENDATIONS", {})
        gemini_rows = build_external_source_rows("gemini", gemini_dict, eligible_samples)
        print("Loaded gemini rows:", len(gemini_rows))
        all_results.append(pd.DataFrame(gemini_rows))

combined_exp_df = pd.concat([x for x in all_results if not x.empty], ignore_index=True)

if combined_exp_df.empty:
    raise ValueError("No recommendation rows were loaded from any source.")

print("Loaded combined experiment rows:", len(combined_exp_df))


# -----------------------
# 5) MERGE WITH GROUND TRUTH CSV
# -----------------------
merged = df.merge(combined_exp_df, on="sample_norm", how="inner", suffixes=("_csv", "_exp"))
merged["sample"] = merged["sample_exp"]
merged["user_intent"] = merged["sample_norm"].map(intent_map)

print("\nCSV samples:")
print(sorted(df["sample_norm"].dropna().astype(str).unique().tolist()))

print("\nCombined source samples:")
print(sorted(combined_exp_df["sample_norm"].dropna().astype(str).unique().tolist()))

merged = merged[
    merged["recommendations"].apply(lambda x: isinstance(x, list) and len(x) > 0)
].copy()

print("Rows after merging experiments with samples:", len(merged))

if merged.empty:
    raise ValueError("Merged dataframe is empty. Check sample names, paths, and external recommendation dicts.")


# -----------------------
# 6) COMPUTE EXACT MODEL METRICS
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
# 7) COMPUTE ATTRIBUTE-LEVEL MATCH METRICS
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
# 8) DETAIL OUTPUT
# -----------------------
full_detail_cols = [
    "source",
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

detail_df = merged[full_detail_cols].copy()

compact_detail_cols = [
    "source",
    "sample",
    "experiment",
    "paper",
    "title",
    "modelID",
    "year",
    "venue",
    "num_recommendations",
    "rank",
    "accuracy@1",
    f"hit@{K}",
    f"ndcg@{K}",
    "gt_family_root",
    "family_root_first_rank",
    f"family_root_hit@{K}",
    "gt_assigned_modality",
    "assigned_modality_first_rank",
    f"assigned_modality_hit@{K}",
    "gt_task",
    "task_first_rank",
    f"task_hit@{K}",
    "response_path",
]
compact_detail_df = detail_df[compact_detail_cols].copy()


# -----------------------
# 9) SUMMARY BY SOURCE + EXPERIMENT
# -----------------------
experiment_summary_rows = []
for (source, experiment), group in detail_df.groupby(["source", "experiment"]):
    stats = summarize_group(group, k=K, unique_sample_col="sample")
    stats["source"] = source
    stats["experiment"] = experiment
    experiment_summary_rows.append(stats)

experiment_summary_df = pd.DataFrame(experiment_summary_rows)

experiment_summary_df = experiment_summary_df[
    [
        "source",
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
].sort_values(by=["source", "experiment"])


# -----------------------
# 10) SUMMARY BY SAMPLE + SOURCE
# -----------------------
sample_summary_rows = []
for (sample, source), group in detail_df.groupby(["sample", "source"]):
    stats = summarize_group(group, k=K)
    stats["sample"] = sample
    stats["source"] = source
    stats["num_experiments"] = group["experiment"].nunique()

    gt_model = group["modelID"].iloc[0] if len(group) > 0 else None
    gt_attrs = get_model_attrs(gt_model, model_meta)
    stats["gt_modelID"] = gt_model if gt_model else NO_INFO
    stats["gt_family_root"] = value_or_no_info(gt_attrs.get("family_root"))
    stats["gt_assigned_modality"] = value_or_no_info(gt_attrs.get("assigned_modality"))
    stats["gt_task"] = value_or_no_info(gt_attrs.get("task"))

    sample_summary_rows.append(stats)

sample_summary_df = pd.DataFrame(sample_summary_rows)

sample_summary_df = sample_summary_df[
    [
        "sample",
        "source",
        "gt_modelID",
        "gt_family_root",
        "gt_assigned_modality",
        "gt_task",
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
].sort_values(by=["sample", "source"])


# -----------------------
# 11) SAVE
# -----------------------
os.makedirs(OUT_DIR, exist_ok=True)

if DETAIL_OUTPUT_MODE in {"full", "both"}:
    detail_df.to_csv(OUT_DETAIL_CSV, index=False, encoding="utf-8")
    print(f"Saved full detail CSV to: {OUT_DETAIL_CSV}")

if DETAIL_OUTPUT_MODE in {"compact", "both"}:
    compact_detail_df.to_csv(OUT_COMPACT_DETAIL_CSV, index=False, encoding="utf-8")
    print(f"Saved compact detail CSV to: {OUT_COMPACT_DETAIL_CSV}")

experiment_summary_df.to_csv(OUT_EXPERIMENT_CSV, index=False, encoding="utf-8")
sample_summary_df.to_csv(OUT_SAMPLE_CSV, index=False, encoding="utf-8")

print(f"Saved experiment summary CSV to: {OUT_EXPERIMENT_CSV}")
print(f"Saved sample summary CSV to: {OUT_SAMPLE_CSV}")
