import pandas as pd
import json
import ast
import numpy as np

ORIGINAL_PATH = "10-EVALUATION/enriched_ffs/enriched_ffs_eval_T5_united.csv"
OUT_PATH      = "10-EVALUATION/enriched_ffs/enriched_ffs_eval_T5_filtered.csv"

# WORST_PREDICTED_FP_JSON = "10-EVALUATION/results_samples/worst_predicted_fp_T5.json"
# WORST_TRUTH_FN_JSON     = "10-EVALUATION/results_samples/worst_truth_fn_T5.json"

WORST_PREDICTED_FP_JSON = "10-EVALUATION/results_samples/balanced_fp_remove_T5.json"
WORST_TRUTH_FN_JSON     = "10-EVALUATION/results_samples/balanced_fn_add_T5.json"
# FAST_RULES = "10-Evaluation/results_samples/fast_rules.json"

df = pd.read_csv(ORIGINAL_PATH)
# Keep only first 50 rows
KAC = 100
df_first_50 = df.head(KAC)
df_first_50.to_csv("10-EVALUATION/enriched_ffs/enriched_ffs_eval_T5_united_50.csv", index=False)

df0 = df.copy()

with open(WORST_PREDICTED_FP_JSON, "r") as f:
    worst_predicted = json.load(f)

with open(WORST_TRUTH_FN_JSON, "r") as f:
    worst_truth = json.load(f)

worst_predicted_features = {item.get("feature") for item in worst_predicted if isinstance(item, dict) and item.get("feature") is not None}
worst_truth_features     = {item.get("feature") for item in worst_truth if isinstance(item, dict) and item.get("feature") is not None}

LIST_COLS = ["Pipeline_Features", "Gemini_Features", "ChatGPT_Features", "Instruct_Features", "United_Features"]

def to_list(x):
    """Convert CSV cell to python list safely."""
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return []
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []

def filter_list(lst, banned_set):
    lst = to_list(lst)
    return [v for v in lst if v not in banned_set]

# Parse list columns first
for c in LIST_COLS:
    if c in df.columns:
        df[c] = df[c].apply(to_list)

# Apply filtering rules (edit mapping if your intent differs)
if "Pipeline_Features" in df.columns:
    df["Pipeline_Features"] = df["Pipeline_Features"].apply(lambda lst: filter_list(lst, worst_predicted_features))

for c in ["Gemini_Features", "ChatGPT_Features", "Instruct_Features", "United_Features"]:
    if c in df.columns:
        df[c] = df[c].apply(lambda lst: filter_list(lst, worst_truth_features))

# Counts with explicit names
COUNT_MAP = {
    "Pipeline_Features": "Pipe_count",
    "Instruct_Features": "Inst_count",
    "ChatGPT_Features": "Chat_count",
    "Gemini_Features": "Gemi_count",
    "United_Features": "Unit_count",
}
for feat_col, count_col in COUNT_MAP.items():
    if feat_col in df.columns:
        df[count_col] = df[feat_col].apply(len)

# Keep only available columns (prevents KeyError)
wanted = [
    "ModelID",
    "Pipeline_Features", "Pipe_count",
    "Instruct_Features", "Inst_count",
    "ChatGPT_Features", "Chat_count",
    "Gemini_Features", "Gemi_count",
    "United_Features", "Unit_count",
    "Descriptions", "Description_norm",
]
df = df[[c for c in wanted if c in df.columns]]

df.to_csv(OUT_PATH, index=False)
print(f"Wrote: {OUT_PATH}")

df2 = df.copy()
df2 = df2.head(KAC)
df2.to_csv("10-EVALUATION/enriched_ffs/enriched_ffs_eval_T5_filtered_50.csv", index=False)

import ast

cols = ["Pipeline_Features","Gemini_Features","ChatGPT_Features","Instruct_Features","United_Features"]

# if your columns are strings like "['a','b']", convert them to lists
for c in cols:
    if c in df0.columns:
        df0[c] = df0[c].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else []))
    if c in df.columns:
        df[c]  = df[c].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else []))

print("\nFeature totals (before -> after):")
for c in cols:
    if c in df0.columns and c in df.columns:
        before = df0[c].apply(len).sum()
        after  = df[c].apply(len).sum()
        print(f"{c:18s} {before:7d} -> {after:7d}  (removed {before-after})")

