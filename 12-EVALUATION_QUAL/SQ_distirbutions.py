import pandas as pd
import numpy as np
import json

df = pd.read_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/sentiment_for_f5_united_3.csv")

total_sample_size = 500
min_per_class = 100

# -------------------------------
# ORIGINAL DISTRIBUTION
# -------------------------------
print("\nORIGINAL DISTRIBUTION")
print(df["mode_2"].value_counts())
print(df["mode_2"].value_counts(normalize=True))

# Count classes
counts = df["mode_2"].value_counts()

# Start with minimum per class
sample_sizes = {label: min_per_class for label in counts.index}

remaining = total_sample_size - sum(sample_sizes.values())

# Allocate remaining proportionally
proportions = counts / counts.sum()
additional = (proportions * remaining).round().astype(int)

for label in sample_sizes:
    sample_sizes[label] += additional[label]

# Adjust if off by rounding
diff = total_sample_size - sum(sample_sizes.values())
if diff != 0:
    largest_class = counts.idxmax()
    sample_sizes[largest_class] += diff

print("\nTARGET SAMPLE ALLOCATION")
print(sample_sizes)

# -------------------------------
# SAMPLING
# -------------------------------
sampled_df = (
    df.groupby("mode_2", group_keys=False)
      .apply(lambda x: x.sample(n=min(sample_sizes[x.name], len(x)), random_state=42))
)

sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------------
# SAMPLED DISTRIBUTION
# -------------------------------
print("\nSAMPLED DISTRIBUTION")
print(sampled_df["mode_2"].value_counts())
print(sampled_df["mode_2"].value_counts(normalize=True))

# Save sample
sampled_df.to_csv("12-EVALUATION_QUAL/sentiment_sample_aaa.csv", index=False)


# ============================================================
# RESTORE ORIGINAL TEXT
# ============================================================

IN_CSV = "12-EVALUATION_QUAL/sentiment_sample_aaa.csv"
MAPPING_JSON = "6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/preprocessing_sentiment.json"
OUT_CSV = "12-EVALUATION_QUAL/sentiment_sample_2_aaa.csv"

df = pd.read_csv(IN_CSV)

with open(MAPPING_JSON, "r", encoding="utf-8") as f:
    mapping = json.load(f)

map_df = pd.DataFrame(mapping)

map_df = map_df[["model_id", "processed", "original"]].dropna()
map_df = map_df.drop_duplicates(subset=["model_id", "processed"], keep="first")

PROCESSED_COL = "reviews"

if PROCESSED_COL not in df.columns:
    raise ValueError(f"Couldn't find '{PROCESSED_COL}' in CSV columns: {list(df.columns)}")

out = df.merge(
    map_df,
    how="left",
    left_on=["model_id", PROCESSED_COL],
    right_on=["model_id", "processed"],
)

out = out.drop(columns=["processed"])

# -------------------------------
# MATCH RATE
# -------------------------------
missing = out["original"].isna().sum()
print(f"\nRows: {len(out)} | Missing originals: {missing}")

# -------------------------------
# FINAL DISTRIBUTION
# -------------------------------
print("\nFINAL DISTRIBUTION (after merge)")
print(out["mode_2"].value_counts())
print(out["mode_2"].value_counts(normalize=True))

out.to_csv(OUT_CSV, index=False)
print(f"\nWrote: {OUT_CSV}")