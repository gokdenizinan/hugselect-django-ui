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

DETAIL_OUT = os.path.join(OUT_DIR, "experiment_grid_accuracy_detail.csv")
VALUE_EFFECTS_OUT = os.path.join(OUT_DIR, "grid_value_effects.csv")
PARAM_IMPORTANCE_OUT = os.path.join(OUT_DIR, "grid_parameter_importance.csv")

# Only include these samples
ALLOWED_SAMPLES = {"A55", "A62", "A90", "A98", "A99", "A130", "A131", "A158"}
# Optional:
# If True, also pull a few non-grid config fields into the detail table
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

# =========================================================
# HELPERS
# =========================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_experiment_path(path, run_name):
    """
    Expected path pattern:
      .../8-CRITERIA_SELECTION/experiments/experiment_runs_H/A130/exp_006/summary.json

    Returns:
      sample='A130', experiment='exp_006'
    """
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
    """
    Convert correct_rank into numeric rank.
    If missing / null, keep NaN.
    """
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def flatten_summary(summary, sample_from_path=None, experiment_from_path=None):
    """
    Extract one row from a summary.json file.
    """
    config = summary.get("config", {}) or {}
    grid_values = config.get("grid_values", {}) or {}

    row = {
        "eval_id": summary.get("eval_id"),
        "sample_key": summary.get("sample_key"),
        "sample": sample_from_path or summary.get("eval_id"),
        "experiment_id": summary.get("experiment_id") or experiment_from_path,
        "correct_model": summary.get("correct_model"),
        "top1_model": summary.get("top1_model"),
        "correct_rank": normalize_rank(summary.get("correct_rank")),
        "hit_at_1": bool(summary.get("hit_at_1", False)),
        "hit_at_10": bool(summary.get("hit_at_10", False)),
        "elapsed_seconds": safe_float(summary.get("elapsed_seconds")),
    }

    # Add stage timings if present
    timings = summary.get("stage_timings", {}) or {}
    for k, v in timings.items():
        row[f"timing__{k}"] = safe_float(v)

    # Add grid values
    for k, v in grid_values.items():
        row[f"grid__{k}"] = v

    # Optionally add a few extra config fields
    if INCLUDE_EXTRA_CONFIG_FIELDS:
        for k in EXTRA_CONFIG_FIELDS:
            if k in config:
                row[f"config__{k}"] = config.get(k)

    return row

def compute_rank_score(rank, k=10):
    """
    Optional continuous score derived from rank.
    Higher is better, 0 if not in top-k.
    """
    if pd.isna(rank) or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)

def weighted_std_of_group_means(group_df, mean_col, weight_col):
    """
    Weighted std deviation of group means around the global mean.
    Useful as a simple 'importance' signal for each parameter.
    """
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

# Add numeric forms
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

global_hit1 = detail_df["hit_at_1_num"].mean()
global_hit10 = detail_df["hit_at_10_num"].mean()
global_rank = detail_df["correct_rank"].dropna().mean() if detail_df["correct_rank"].notna().any() else np.nan
global_rank_score = detail_df["rank_score@10"].mean()

value_effect_rows = []

for param_col in grid_cols:
    param_name = param_col.replace("grid__", "", 1)

    sub = detail_df[[param_col, "hit_at_1_num", "hit_at_10_num", "correct_rank", "rank_score@10"]].copy()
    sub = sub.rename(columns={param_col: "param_value"})

    # Keep missing values visible if they exist
    if sub["param_value"].isna().any():
        # Convert NaNs to string token for grouping clarity
        sub["param_value_group"] = sub["param_value"].astype(object).where(sub["param_value"].notna(), "__MISSING__")
    else:
        sub["param_value_group"] = sub["param_value"]

    grouped = sub.groupby("param_value_group", dropna=False)

    for value, g in grouped:
        ranks = g["correct_rank"].dropna()

        row = {
            "parameter": param_name,
            "value": value,
            "num_experiments": int(len(g)),
            "hit_at_1_rate": round(g["hit_at_1_num"].mean(), 6),
            "hit_at_10_rate": round(g["hit_at_10_num"].mean(), 6),
            "mean_rank": round(ranks.mean(), 6) if len(ranks) > 0 else np.nan,
            "median_rank": round(ranks.median(), 6) if len(ranks) > 0 else np.nan,
            "rank_score@10_mean": round(g["rank_score@10"].mean(), 6),

            # Deltas from overall baseline
            "delta_hit_at_1_vs_global": round(g["hit_at_1_num"].mean() - global_hit1, 6),
            "delta_hit_at_10_vs_global": round(g["hit_at_10_num"].mean() - global_hit10, 6),
            "delta_rank_score@10_vs_global": round(g["rank_score@10"].mean() - global_rank_score, 6),
        }
        value_effect_rows.append(row)

value_effects_df = pd.DataFrame(value_effect_rows)

# Sort by strongest positive hit@10 effect first
value_effects_df = value_effects_df.sort_values(
    by=["parameter", "delta_hit_at_10_vs_global", "hit_at_10_rate"],
    ascending=[True, False, False]
)

value_effects_df.to_csv(VALUE_EFFECTS_OUT, index=False, encoding="utf-8")

# =========================================================
# 4) BUILD PARAMETER-LEVEL IMPORTANCE TABLE
# =========================================================
param_importance_rows = []

for param_name, g in value_effects_df.groupby("parameter"):
    # Skip parameters with only one observed value
    n_values = g["value"].nunique(dropna=False)

    best_hit10_row = g.sort_values(
        by=["hit_at_10_rate", "num_experiments"],
        ascending=[False, False]
    ).iloc[0]

    worst_hit10_row = g.sort_values(
        by=["hit_at_10_rate", "num_experiments"],
        ascending=[True, False]
    ).iloc[0]

    best_rank_score_row = g.sort_values(
        by=["rank_score@10_mean", "num_experiments"],
        ascending=[False, False]
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

        # Simple effect-size style indicators
        "importance_hit_at_10": round(float(importance_hit10), 6) if not pd.isna(importance_hit10) else np.nan,
        "importance_hit_at_1": round(float(importance_hit1), 6) if not pd.isna(importance_hit1) else np.nan,
        "importance_rank_score@10": round(float(importance_rank_score), 6) if not pd.isna(importance_rank_score) else np.nan,
    }

    param_importance_rows.append(row)

param_importance_df = pd.DataFrame(param_importance_rows)

# Main ranking: which parameter changes performance the most on hit@10
param_importance_df = param_importance_df.sort_values(
    by=["importance_hit_at_10", "delta_best_vs_worst_hit_at_10"],
    ascending=[False, False]
)

param_importance_df.to_csv(PARAM_IMPORTANCE_OUT, index=False, encoding="utf-8")

# =========================================================
# 5) PRINT QUICK SUMMARY
# =========================================================
print(f"Loaded experiment summaries: {len(detail_df)}")
print(f"Saved detail CSV: {DETAIL_OUT}")
print(f"Saved per-value effects CSV: {VALUE_EFFECTS_OUT}")
print(f"Saved parameter importance CSV: {PARAM_IMPORTANCE_OUT}")

print("\nTop parameters by estimated impact on hit@10:")
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
                "importance_hit_at_10",
            ]
        ].head(15).to_string(index=False)
    )
else:
    print("No parameter importance rows were produced.")