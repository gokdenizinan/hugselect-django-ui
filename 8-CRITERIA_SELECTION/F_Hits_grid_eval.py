import os
import json
import glob
import math
import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================
RUN = "experiment_runs_H"
EXPERIMENT_ROOT = f"8-CRITERIA_SELECTION/experiments/{RUN}"
OUT_DIR = f"8-CRITERIA_SELECTION/hits_cluster/{RUN}/GRID_EVAL"
os.makedirs(OUT_DIR, exist_ok=True)
DETAIL_OUT = os.path.join(OUT_DIR, "experiment_grid_accuracy_detail.csv")
VALUE_EFFECTS_OUT = os.path.join(OUT_DIR, "grid_value_effects.csv")
PARAM_IMPORTANCE_OUT = os.path.join(OUT_DIR, "grid_parameter_importance.csv")

# ALLOWED_SAMPLES = {"A55", "A62", "A90", "A98", "A99", "A130", "A131", "A158"}
ALLOWED_SAMPLES = None
INCLUDE_EXTRA_CONFIG_FIELDS = True

EXTRA_CONFIG_FIELDS = [
    "hybrid_search_enabled",
    "bm25_rrf_weight",
    "vector_rrf_weight",
    "vector_top_k",
    "vector_num_candidates",
    "minimum_should_match",
    "synonym_min_conf",
    "target_hits",
    "size",
]

USE_LEAVE_ONE_OUT_BASELINE = True

# =========================================================
# HELPERS
# =========================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_experiment_path(path, run_name):
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)

    if run_name not in parts:
        return None, None

    idx = parts.index(run_name)
    if len(parts) <= idx + 2:
        return None, None

    sample = parts[idx + 1]
    experiment = parts[idx + 2]
    return sample, experiment

def safe_float(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def normalize_rank(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def normalize_model_name(name):
    """
    Normalize model identifiers so comparisons are stable.

    Examples:
      Helsinki-NLP__opus-mt-mul-en.json -> Helsinki-NLP__opus-mt-mul-en
      google/mt5-base -> google/mt5-base
      model.json -> model
    """
    if name is None:
        return None

    s = str(name).strip()
    if not s:
        return None

    # only strip a final .json suffix
    if s.lower().endswith(".json"):
        s = s[:-5]

    return s

def extract_top10_models(summary):
    """
    Try a few common locations for the ranked model list.
    Returns a normalized list of model names.
    """
    candidates = []

    for key in ["top_10_models", "top10_models", "top_models"]:
        val = summary.get(key)
        if isinstance(val, list):
            candidates = val
            break

    normalized = []
    for x in candidates:
        if isinstance(x, str):
            normalized.append(normalize_model_name(x))
        elif isinstance(x, dict):
            # handle shapes like {"model": "..."} or {"model_name": "..."}
            model_name = x.get("model") or x.get("model_name") or x.get("name")
            normalized.append(normalize_model_name(model_name))
        else:
            normalized.append(normalize_model_name(x))

    return [x for x in normalized if x is not None]

def recompute_hits_and_rank(summary):
    """
    Recompute hit_at_1 / hit_at_10 / correct_rank from normalized names when possible.
    Falls back to summary values if no top-10 list exists.
    """
    correct_model_raw = summary.get("correct_model")
    correct_model = normalize_model_name(correct_model_raw)

    top_10_models = extract_top10_models(summary)

    if correct_model and top_10_models:
        hit_at_1 = len(top_10_models) > 0 and top_10_models[0] == correct_model
        hit_at_10 = correct_model in top_10_models

        if hit_at_10:
            correct_rank = float(top_10_models.index(correct_model) + 1)
        else:
            correct_rank = np.nan

        top1_model = top_10_models[0] if len(top_10_models) > 0 else None

        return {
            "correct_model_normalized": correct_model,
            "top1_model_normalized": top1_model,
            "top_10_models_normalized": top_10_models,
            "correct_rank": correct_rank,
            "hit_at_1": hit_at_1,
            "hit_at_10": hit_at_10,
            "normalization_recomputed": True,
        }

    return {
        "correct_model_normalized": correct_model,
        "top1_model_normalized": normalize_model_name(summary.get("top1_model")),
        "top_10_models_normalized": top_10_models,
        "correct_rank": normalize_rank(summary.get("correct_rank")),
        "hit_at_1": bool(summary.get("hit_at_1", False)),
        "hit_at_10": bool(summary.get("hit_at_10", False)),
        "normalization_recomputed": False,
    }

def flatten_summary(summary, sample_from_path=None, experiment_from_path=None):
    config = summary.get("config", {}) or {}
    grid_values = config.get("grid_values", {}) or {}

    recomputed = recompute_hits_and_rank(summary)

    row = {
        "eval_id": summary.get("eval_id"),
        "sample_key": summary.get("sample_key"),
        "sample": sample_from_path or summary.get("eval_id"),
        "experiment_id": summary.get("experiment_id") or experiment_from_path,

        "correct_model": summary.get("correct_model"),
        "correct_model_normalized": recomputed["correct_model_normalized"],

        "top1_model": summary.get("top1_model"),
        "top1_model_normalized": recomputed["top1_model_normalized"],

        "correct_rank": recomputed["correct_rank"],
        "hit_at_1": recomputed["hit_at_1"],
        "hit_at_10": recomputed["hit_at_10"],

        "normalization_recomputed": recomputed["normalization_recomputed"],
        "top_10_models_normalized": json.dumps(recomputed["top_10_models_normalized"], ensure_ascii=False),

        "elapsed_seconds": safe_float(summary.get("elapsed_seconds")),
    }

    timings = summary.get("stage_timings", {}) or {}
    for k, v in timings.items():
        row[f"timing__{k}"] = safe_float(v)

    for k, v in grid_values.items():
        row[f"grid__{k}"] = v

    if INCLUDE_EXTRA_CONFIG_FIELDS:
        for k in EXTRA_CONFIG_FIELDS:
            if k in config:
                row[f"config__{k}"] = config.get(k)

    return row

def compute_rank_score(rank, k=10):
    if pd.isna(rank) or rank > k or rank < 1:
        return 0.0
    return 1.0 / math.log2(rank + 1)

def weighted_std_of_group_means(group_df, mean_col, weight_col):
    if len(group_df) == 0:
        return np.nan

    means = group_df[mean_col].astype(float)
    weights = group_df[weight_col].astype(float)

    total_weight = weights.sum()
    if total_weight <= 0:
        return np.nan

    weighted_mean = np.average(means, weights=weights)
    weighted_var = np.average((means - weighted_mean) ** 2, weights=weights)
    return math.sqrt(weighted_var)

def make_groupable_value_series(series):
    return series.astype(object).where(series.notna(), "__MISSING__")

def compute_other_group_mean(full_df, param_col, metric_col):
    values = make_groupable_value_series(full_df[param_col])
    result = {}

    distinct_values = pd.Series(values).drop_duplicates().tolist()
    for value in distinct_values:
        mask_this = values == value
        mask_other = ~mask_this

        other_metric = full_df.loc[mask_other, metric_col]
        if len(other_metric) == 0:
            result[value] = np.nan
        else:
            result[value] = float(other_metric.mean())

    return result

# =========================================================
# 1) LOAD ALL summary.json FILES
# =========================================================
pattern = os.path.join(EXPERIMENT_ROOT, "*", "*", "summary.json")
summary_paths = sorted(glob.glob(pattern))

if not summary_paths:
    raise ValueError(f"No summary.json files found under: {EXPERIMENT_ROOT}")

rows = []

for path in summary_paths:
    sample, experiment = parse_experiment_path(path, RUN)

    try:
        summary = load_json(path)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        continue

    row = flatten_summary(summary, sample_from_path=sample, experiment_from_path=experiment)
    row["summary_path"] = path
    rows.append(row)

detail_df = pd.DataFrame(rows)

if ALLOWED_SAMPLES:
    detail_df = detail_df[detail_df["sample"].isin(ALLOWED_SAMPLES)].copy()

if detail_df.empty:
    raise ValueError("No valid summary rows were loaded.")

detail_df["hit_at_1_num"] = detail_df["hit_at_1"].astype(int)
detail_df["hit_at_10_num"] = detail_df["hit_at_10"].astype(int)
detail_df["rank_score@10"] = detail_df["correct_rank"].apply(lambda r: compute_rank_score(r, k=10))

# =========================================================
# 2) SAVE DETAIL TABLE
# =========================================================
os.makedirs(OUT_DIR, exist_ok=True)
detail_df.to_csv(DETAIL_OUT, index=False, encoding="utf-8")

# =========================================================
# 3) BUILD PER-VALUE EFFECT TABLE
# =========================================================
grid_cols = [c for c in detail_df.columns if c.startswith("grid__")]

if not grid_cols:
    raise ValueError("No grid values found in config.grid_values.")

global_hit1 = float(detail_df["hit_at_1_num"].mean())
global_hit10 = float(detail_df["hit_at_10_num"].mean())
global_rank_score = float(detail_df["rank_score@10"].mean())

value_effect_rows = []

for param_col in grid_cols:
    param_name = param_col.replace("grid__", "", 1)

    sub = detail_df[[param_col, "hit_at_1_num", "hit_at_10_num", "correct_rank", "rank_score@10"]].copy()
    sub = sub.rename(columns={param_col: "param_value"})
    sub["param_value_group"] = make_groupable_value_series(sub["param_value"])

    grouped = sub.groupby("param_value_group", dropna=False)

    if USE_LEAVE_ONE_OUT_BASELINE:
        other_hit1_means = compute_other_group_mean(detail_df, param_col, "hit_at_1_num")
        other_hit10_means = compute_other_group_mean(detail_df, param_col, "hit_at_10_num")
        other_rank_score_means = compute_other_group_mean(detail_df, param_col, "rank_score@10")
    else:
        other_hit1_means = {}
        other_hit10_means = {}
        other_rank_score_means = {}

    for value, g in grouped:
        ranks = g["correct_rank"].dropna()

        hit1_mean = float(g["hit_at_1_num"].mean())
        hit10_mean = float(g["hit_at_10_num"].mean())
        rank_score_mean = float(g["rank_score@10"].mean())

        if USE_LEAVE_ONE_OUT_BASELINE:
            baseline_hit1 = other_hit1_means.get(value, np.nan)
            baseline_hit10 = other_hit10_means.get(value, np.nan)
            baseline_rank_score = other_rank_score_means.get(value, np.nan)
        else:
            baseline_hit1 = global_hit1
            baseline_hit10 = global_hit10
            baseline_rank_score = global_rank_score

        row = {
            "parameter": param_name,
            "value": value,
            "num_experiments": int(len(g)),
            "hit_at_1_rate": round(hit1_mean, 6),
            "hit_at_10_rate": round(hit10_mean, 6),
            "mean_rank": round(ranks.mean(), 6) if len(ranks) > 0 else np.nan,
            "median_rank": round(ranks.median(), 6) if len(ranks) > 0 else np.nan,
            "rank_score@10_mean": round(rank_score_mean, 6),

            "delta_hit_at_1_vs_global": round(hit1_mean - global_hit1, 6),
            "delta_hit_at_10_vs_global": round(hit10_mean - global_hit10, 6),
            "delta_rank_score@10_vs_global": round(rank_score_mean - global_rank_score, 6),

            "delta_hit_at_1_vs_other_values": round(hit1_mean - baseline_hit1, 6) if not pd.isna(baseline_hit1) else np.nan,
            "delta_hit_at_10_vs_other_values": round(hit10_mean - baseline_hit10, 6) if not pd.isna(baseline_hit10) else np.nan,
            "delta_rank_score@10_vs_other_values": round(rank_score_mean - baseline_rank_score, 6) if not pd.isna(baseline_rank_score) else np.nan,
        }
        value_effect_rows.append(row)

value_effects_df = pd.DataFrame(value_effect_rows)

sort_delta_col = "delta_hit_at_10_vs_other_values" if USE_LEAVE_ONE_OUT_BASELINE else "delta_hit_at_10_vs_global"
value_effects_df = value_effects_df.sort_values(
    by=["parameter", sort_delta_col, "rank_score@10_mean", "hit_at_10_rate"],
    ascending=[True, False, False, False]
)

value_effects_df.to_csv(VALUE_EFFECTS_OUT, index=False, encoding="utf-8")

# =========================================================
# 4) BUILD PARAMETER-LEVEL IMPORTANCE TABLE
# =========================================================
param_importance_rows = []

for param_name, g in value_effects_df.groupby("parameter"):
    g = g.copy()
    n_values = g["value"].nunique(dropna=False)

    if n_values < 2:
        continue

    best_hit10_row = g.sort_values(
        by=["hit_at_10_rate", "rank_score@10_mean", "num_experiments"],
        ascending=[False, False, False]
    ).iloc[0]

    worst_hit10_row = g.sort_values(
        by=["hit_at_10_rate", "rank_score@10_mean", "num_experiments"],
        ascending=[True, True, False]
    ).iloc[0]

    best_rank_score_row = g.sort_values(
        by=["rank_score@10_mean", "hit_at_10_rate", "num_experiments"],
        ascending=[False, False, False]
    ).iloc[0]

    importance_hit10 = weighted_std_of_group_means(
        g, mean_col="hit_at_10_rate", weight_col="num_experiments"
    )

    importance_hit1 = weighted_std_of_group_means(
        g, mean_col="hit_at_1_rate", weight_col="num_experiments"
    )

    importance_rank_score = weighted_std_of_group_means(
        g, mean_col="rank_score@10_mean", weight_col="num_experiments"
    )

    row = {
        "parameter": param_name,
        "num_distinct_values": int(n_values),
        "total_experiments": int(g["num_experiments"].sum()),

        "best_value_by_hit_at_10": best_hit10_row["value"],
        "best_hit_at_10_rate": round(float(best_hit10_row["hit_at_10_rate"]), 6),

        "worst_value_by_hit_at_10": worst_hit10_row["value"],
        "worst_hit_at_10_rate": round(float(worst_hit10_row["hit_at_10_rate"]), 6),

        "delta_best_vs_worst_hit_at_10": round(
            float(best_hit10_row["hit_at_10_rate"]) - float(worst_hit10_row["hit_at_10_rate"]),
            6
        ),

        "best_value_by_rank_score@10": best_rank_score_row["value"],
        "best_rank_score@10_mean": round(float(best_rank_score_row["rank_score@10_mean"]), 6),

        "importance_hit_at_10": round(float(importance_hit10), 6) if not pd.isna(importance_hit10) else np.nan,
        "importance_hit_at_1": round(float(importance_hit1), 6) if not pd.isna(importance_hit1) else np.nan,
        "importance_rank_score@10": round(float(importance_rank_score), 6) if not pd.isna(importance_rank_score) else np.nan,

        "spread_hit_at_10": round(float(g["hit_at_10_rate"].max() - g["hit_at_10_rate"].min()), 6),
        "spread_hit_at_1": round(float(g["hit_at_1_rate"].max() - g["hit_at_1_rate"].min()), 6),
        "spread_rank_score@10": round(float(g["rank_score@10_mean"].max() - g["rank_score@10_mean"].min()), 6),

        "max_abs_delta_hit_at_10_vs_other_values": round(float(g["delta_hit_at_10_vs_other_values"].abs().max()), 6),
        "max_abs_delta_rank_score@10_vs_other_values": round(float(g["delta_rank_score@10_vs_other_values"].abs().max()), 6),
    }

    param_importance_rows.append(row)

param_importance_df = pd.DataFrame(param_importance_rows)

if not param_importance_df.empty:
    param_importance_df = param_importance_df.sort_values(
        by=[
            "spread_rank_score@10",
            "importance_rank_score@10",
            "spread_hit_at_10",
            "delta_best_vs_worst_hit_at_10",
        ],
        ascending=[False, False, False, False]
    )

param_importance_df.to_csv(PARAM_IMPORTANCE_OUT, index=False, encoding="utf-8")

# =========================================================
# 5) PRINT QUICK SUMMARY / DIAGNOSTICS
# =========================================================
print(f"Loaded experiment summaries: {len(detail_df)}")
print(f"Saved detail CSV: {DETAIL_OUT}")
print(f"Saved per-value effects CSV: {VALUE_EFFECTS_OUT}")
print(f"Saved parameter importance CSV: {PARAM_IMPORTANCE_OUT}")

print("\nGlobal metrics:")
print(f"  hit@1 mean: {detail_df['hit_at_1_num'].mean():.6f}")
print(f"  hit@10 mean: {detail_df['hit_at_10_num'].mean():.6f}")
print(f"  rank_score@10 mean: {detail_df['rank_score@10'].mean():.6f}")

print("\nNormalization diagnostics:")
print(f"  rows recomputed from normalized top_10_models: {int(detail_df['normalization_recomputed'].sum())}")
print(f"  rows total: {len(detail_df)}")

print("\nGrid columns found:")
for col in grid_cols:
    distinct_n = detail_df[col].astype(object).where(detail_df[col].notna(), "__MISSING__").nunique(dropna=False)
    print(f"  {col}: {distinct_n} distinct values")

print("\nTop parameters by estimated impact:")
if len(param_importance_df) > 0:
    print(
        param_importance_df[
            [
                "parameter",
                "num_distinct_values",
                "best_value_by_hit_at_10",
                "best_hit_at_10_rate",
                "worst_value_by_hit_at_10",
                "worst_hit_at_10_rate",
                "delta_best_vs_worst_hit_at_10",
                "spread_rank_score@10",
                "importance_rank_score@10",
            ]
        ].head(15).to_string(index=False)
    )
else:
    print("No parameter importance rows were produced.")