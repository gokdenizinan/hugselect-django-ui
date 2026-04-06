import json
import glob
import os
import math
import pandas as pd
import numpy as np
import re
import importlib.util

# -----------------------
# CONFIG
# -----------------------
CSV_PATH = "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/merged_2.csv"
FILTER_JSON_PATH = "11-RECOMMENDATION_EVALUATION/OUTPUT_F.json"

RUN = "experiment_runs_H"
EXPERIMENT_ROOT = f"8-CRITERIA_SELECTION/experiments/{RUN}"

# Folder containing model metadata dictionaries
MODEL_META_DIR = "HF-Models-T7-U"

# External recommendation python file
EXTERNAL_RECOMMENDATIONS_PY = "8-CRITERIA_SELECTION/F_Hits_United.py"

OUT_DIR = f"8-CRITERIA_SELECTION/hits_cluster/{RUN}/multisource"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DETAIL_CSV = os.path.join(OUT_DIR, "experiment_sample_stats.csv")
OUT_COMPACT_CSV = os.path.join(OUT_DIR, "experiment_sample_stats_compact.csv")
OUT_EXPERIMENT_CSV = os.path.join(OUT_DIR, "experiment_stats_summary.csv")
OUT_SAMPLE_CSV = os.path.join(OUT_DIR, "sample_stats_summary.csv")

K = 10
KEEP_COLS = ["sample", "paper", "title", "modelID", "year", "venue"]
ATTR_KEYS = ["family_root", "assigned_modality", "task"]
NO_INFO = "NO INFO"

ENABLE_SOURCES = {
    "recsys": True,
    "chatgpt": True,
    "gemini": True,
}

# Metrics to remove from compact output across exact / family / modality / task
DROP_METRIC_SUFFIXES = [
    "accuracy@1",
    f"precision@{K}",
    f"recall@{K}",
    f"ndcg@{K}",
    f"map@{K}",
    f"hit@{K}",
]


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


def denormalize_sample_key(sample_norm):
    if sample_norm is None:
        return None
    m = re.fullmatch(r"A(\d+)", str(sample_norm), flags=re.IGNORECASE)
    if m:
        return f"sample_{int(m.group(1))}"
    return str(sample_norm)


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


def display_attr_value(x, missing_value=np.nan):
    if x is None:
        return missing_value
    if isinstance(x, float) and np.isnan(x):
        return missing_value
    if isinstance(x, list):
        vals = [str(v) for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return " | ".join(vals) if vals else missing_value
    return str(x)


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
    hit_attrs = [get_model_attrs(rec, model_meta) for rec in (recs if isinstance(recs, list) else [])]

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


def load_external_recommendations(py_path):
    if not os.path.exists(py_path):
        print(f"External recommendation file not found: {py_path}")
        return {}, {}

    spec = importlib.util.spec_from_file_location("f_hits_united_module", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    chatgpt = getattr(module, "CHATGPT_RECOMMENDATIONS", {}) or {}
    gemini = getattr(module, "GEMINI_RECOMMENDATIONS", {}) or {}
    return chatgpt, gemini


def build_external_rows(source_name, rec_dict, valid_sample_norms):
    rows = []
    for raw_sample, recs in rec_dict.items():
        sample_norm = normalize_sample_key(raw_sample)
        if sample_norm not in valid_sample_norms:
            continue
        if not isinstance(recs, list) or len(recs) == 0:
            continue
        rows.append({
            "sample": sample_norm,
            "sample_norm": sample_norm,
            "experiment": source_name,
            "source": source_name,
            "response_path": f"python:{EXTERNAL_RECOMMENDATIONS_PY}:{source_name}",
            "recommendations": recs,
            "num_recommendations": len(recs),
        })
    return pd.DataFrame(rows)


def choose_best_experiment(detail_df):
    recsys_only = detail_df[detail_df["source"] == "recsys"].copy()
    if recsys_only.empty:
        return None

    candidates = []
    for experiment, group in recsys_only.groupby("experiment"):
        non_na = group["family_root_first_rank"].dropna()
        candidates.append({
            "experiment": experiment,
            "family_root_mean_first_rank": non_na.mean() if len(non_na) else np.nan,
            "family_root_median_first_rank": non_na.median() if len(non_na) else np.nan,
            "family_root_hit_count": int(group["family_root_first_rank"].notna().sum()),
            "sample_count": int(group["sample"].nunique()),
        })

    cand_df = pd.DataFrame(candidates)
    cand_df = cand_df.sort_values(
        by=["family_root_mean_first_rank", "family_root_median_first_rank", "family_root_hit_count", "sample_count", "experiment"],
        ascending=[True, True, False, False, True],
        na_position="last",
    )
    return cand_df.iloc[0]["experiment"] if not cand_df.empty else None


def add_no_info_strings(sample_summary_df):
    for col in ["gt_family_root", "gt_assigned_modality", "gt_task"]:
        if col in sample_summary_df.columns:
            sample_summary_df[col] = sample_summary_df[col].fillna(NO_INFO)
            sample_summary_df[col] = sample_summary_df[col].replace("", NO_INFO)
    return sample_summary_df


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
    intent_map = {normalize_sample_key(k): v.get("user_intent") for k, v in filter_data.items()}
    valid_samples = {normalize_sample_key(k) for k in filter_data.keys()}
    print("Samples allowed by filter:", len(valid_samples))
    df = df[df["sample_norm"].isin(valid_samples)].copy()

print("Rows after CSV/filter load:", len(df))

# -----------------------
# 2) LOAD MODEL METADATA
# -----------------------
model_meta = load_model_metadata_folder(MODEL_META_DIR)

# -----------------------
# 3) LOAD RECSYS EXPERIMENTS
# -----------------------
recsys_frames = []
if ENABLE_SOURCES.get("recsys", False):
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
            "source": "recsys",
        })

    exp_df = pd.DataFrame(experiment_rows)
    if not exp_df.empty:
        exp_df["sample_norm"] = exp_df["sample"].apply(normalize_sample_key)
        recsys_frames.append(exp_df)
    print("Loaded recsys experiment rows:", len(exp_df))
else:
    exp_df = pd.DataFrame()

if exp_df.empty and ENABLE_SOURCES.get("recsys", False):
    raise ValueError("No recsys experiment response.json files were found.")

valid_recsys_sample_norms = set(exp_df["sample_norm"].dropna().unique().tolist()) if not exp_df.empty else set()

# -----------------------
# 4) LOAD EXTERNAL RECOMMENDATIONS
# -----------------------
chatgpt_dict, gemini_dict = load_external_recommendations(EXTERNAL_RECOMMENDATIONS_PY)
external_frames = []

if ENABLE_SOURCES.get("chatgpt", False):
    external_frames.append(build_external_rows("chatgpt", chatgpt_dict, valid_recsys_sample_norms))
if ENABLE_SOURCES.get("gemini", False):
    external_frames.append(build_external_rows("gemini", gemini_dict, valid_recsys_sample_norms))

all_eval_frames = recsys_frames + [f for f in external_frames if f is not None and not f.empty]
if not all_eval_frames:
    raise ValueError("No evaluation data found across recsys/chatgpt/gemini.")

all_eval_df = pd.concat(all_eval_frames, ignore_index=True)
all_eval_df = all_eval_df[all_eval_df["recommendations"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
print("Loaded all evaluation rows:", len(all_eval_df))

# -----------------------
# 5) MERGE WITH GROUND TRUTH CSV
# -----------------------
merged = df.merge(all_eval_df, on="sample_norm", how="inner", suffixes=("_csv", "_eval"))
merged["sample"] = merged["sample_eval"]
merged["user_intent"] = merged["sample_norm"].map(intent_map)

print("Rows after merging experiments with samples:", len(merged))
if merged.empty:
    raise ValueError("Merged dataframe is empty. Check sample names and input files.")

# -----------------------
# 6) COMPUTE EXACT MODEL METRICS
# -----------------------
merged["rank"] = merged.apply(lambda row: find_rank(row["modelID"], row["recommendations"]), axis=1)
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
    lambda row: compare_gt_to_recommendations(row["modelID"], row["recommendations"], model_meta, k=K),
    axis=1,
)
attr_df = pd.DataFrame(attr_comparisons.tolist(), index=merged.index)
merged = pd.concat([merged, attr_df], axis=1)

# For readability / downstream outputs
for gt_col in ["gt_family_root", "gt_assigned_modality", "gt_task"]:
    merged[gt_col] = merged[gt_col].apply(display_attr_value)

# -----------------------
# 8) FULL DETAIL OUTPUT
# -----------------------
detail_cols = [
    "sample",
    "sample_norm",
    "source",
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
# 9) SUMMARY BY EXPERIMENT + SOURCE
# -----------------------
experiment_summary_rows = []
for (source, experiment), group in detail_df.groupby(["source", "experiment"]):
    stats = summarize_group(group, k=K, unique_sample_col="sample_norm")
    stats["source"] = source
    stats["experiment"] = experiment
    experiment_summary_rows.append(stats)

experiment_summary_df = pd.DataFrame(experiment_summary_rows)
if not experiment_summary_df.empty:
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
    stats["num_rows"] = len(group)
    stats["num_experiments"] = group["experiment"].nunique()
    stats["paper"] = group["paper"].iloc[0]
    stats["title"] = group["title"].iloc[0]
    stats["modelID"] = group["modelID"].iloc[0]
    stats["year"] = group["year"].iloc[0]
    stats["venue"] = group["venue"].iloc[0]
    stats["gt_family_root"] = group["gt_family_root"].iloc[0]
    stats["gt_assigned_modality"] = group["gt_assigned_modality"].iloc[0]
    stats["gt_task"] = group["gt_task"].iloc[0]
    sample_summary_rows.append(stats)

sample_summary_df = pd.DataFrame(sample_summary_rows)
if not sample_summary_df.empty:
    sample_summary_df = sample_summary_df[
        [
            "sample",
            "source",
            "paper",
            "title",
            "modelID",
            "year",
            "venue",
            "gt_family_root",
            "gt_assigned_modality",
            "gt_task",
            "num_experiments",
            "num_rows",
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
    sample_summary_df = add_no_info_strings(sample_summary_df)

# -----------------------
# 11) BUILD COMPACT SIDE-BY-SIDE OUTPUT
#   - keep only best recsys experiment by overall family_root_first_rank
#   - chatgpt and gemini become their own columns
#   - drop accuracy/precision/recall/ndcg/map/hit for exact + family + modality + task
# -----------------------
best_experiment = choose_best_experiment(detail_df)
print("Best recsys experiment by family_root_first_rank overall:", best_experiment)

compact_source_frames = []
base_keep_cols = [
    "sample",
    "paper",
    "title",
    "modelID",
    "year",
    "venue",
    "gt_family_root",
    "gt_assigned_modality",
    "gt_task",
]

source_metric_cols = [
    "rank",
    "num_recommendations",
    "family_root_match_rate",
    "family_root_first_rank",
    "assigned_modality_match_rate",
    "assigned_modality_first_rank",
    "task_match_rate",
    "task_first_rank",
]

if best_experiment is not None:
    recsys_best_df = detail_df[(detail_df["source"] == "recsys") & (detail_df["experiment"] == best_experiment)].copy()
    if not recsys_best_df.empty:
        recsys_best_df = recsys_best_df[base_keep_cols + source_metric_cols].copy()
        recsys_best_df = recsys_best_df.rename(columns={c: f"recsys_{c}" for c in source_metric_cols})
        compact_source_frames.append(recsys_best_df)

for source_name in ["chatgpt", "gemini"]:
    src_df = detail_df[detail_df["source"] == source_name].copy()
    if src_df.empty:
        continue
    src_df = src_df.sort_values(by=["sample", "experiment"]).drop_duplicates(subset=["sample"], keep="first")
    src_df = src_df[base_keep_cols + source_metric_cols].copy()
    src_df = src_df.rename(columns={c: f"{source_name}_{c}" for c in source_metric_cols})
    compact_source_frames.append(src_df)

if compact_source_frames:
    compact_df = compact_source_frames[0].copy()
    for next_df in compact_source_frames[1:]:
        compact_df = compact_df.merge(next_df, on=base_keep_cols, how="outer")
else:
    compact_df = pd.DataFrame(columns=base_keep_cols)

for col in ["gt_family_root", "gt_assigned_modality", "gt_task"]:
    if col in compact_df.columns:
        compact_df[col] = compact_df[col].fillna(NO_INFO).replace("", NO_INFO)

compact_df = compact_df.sort_values(by=["sample"]).reset_index(drop=True)

# Reorder compact columns so each stat is grouped side-by-side across recsys/chatgpt/gemini
compact_column_order = [
    "sample",
    "paper",
    "title",
    "modelID",
    "year",
    "venue",
    "gt_family_root",
    "gt_assigned_modality",
    "gt_task",

    "recsys_num_recommendations",
    "chatgpt_num_recommendations",
    "gemini_num_recommendations",

    "recsys_rank",
    "chatgpt_rank",
    "gemini_rank",

    "recsys_family_root_first_rank",
    "chatgpt_family_root_first_rank",
    "gemini_family_root_first_rank",
    "recsys_family_root_match_rate",
    "chatgpt_family_root_match_rate",
    "gemini_family_root_match_rate",

    "recsys_assigned_modality_first_rank",
    "chatgpt_assigned_modality_first_rank",
    "gemini_assigned_modality_first_rank",
    "recsys_assigned_modality_match_rate",
    "chatgpt_assigned_modality_match_rate",
    "gemini_assigned_modality_match_rate",

    "recsys_task_first_rank",
    "chatgpt_task_first_rank",
    "gemini_task_first_rank",
    "recsys_task_match_rate",
    "chatgpt_task_match_rate",
    "gemini_task_match_rate",
]

existing_compact_cols = [c for c in compact_column_order if c in compact_df.columns]
remaining_compact_cols = [c for c in compact_df.columns if c not in existing_compact_cols]
compact_df = compact_df[existing_compact_cols + remaining_compact_cols]

# -----------------------
# 12) SAVE
# -----------------------
os.makedirs(OUT_DIR, exist_ok=True)

detail_df.to_csv(OUT_DETAIL_CSV, index=False, encoding="utf-8")
experiment_summary_df.to_csv(OUT_EXPERIMENT_CSV, index=False, encoding="utf-8")
sample_summary_df.to_csv(OUT_SAMPLE_CSV, index=False, encoding="utf-8")
compact_df.to_csv(OUT_COMPACT_CSV, index=False, encoding="utf-8")

print(f"Saved detail CSV to: {OUT_DETAIL_CSV}")
print(f"Saved compact CSV to: {OUT_COMPACT_CSV}")
print(f"Saved experiment summary CSV to: {OUT_EXPERIMENT_CSV}")
print(f"Saved sample summary CSV to: {OUT_SAMPLE_CSV}")
