import pandas as pd
import numpy as np

df = pd.read_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/sentiment_for_f5_united_3.csv")

total_sample_size = 500
min_per_class = 100

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

print("Sample allocation:", sample_sizes)

sampled_df = (
    df.groupby("mode_2", group_keys=False)
      .apply(lambda x: x.sample(n=sample_sizes[x.name], random_state=42)
                        .assign(mode_2=x.name))
)

sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)


# sampled_df.to_csv("12-EVALUATION_QUAL/sentiment_sample.csv", index=False)

import json
import pandas as pd

IN_CSV = "12-EVALUATION_QUAL/sentiment_sample.csv"
MAPPING_JSON = "6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/preprocessing_sentiment.json"  # <-- change this
OUT_CSV = "12-EVALUATION_QUAL/sentiment_sample_2.csv"

# 1) load sampled/output csv
df = pd.read_csv(IN_CSV)

# 2) load mapping json (list of dicts)
with open(MAPPING_JSON, "r", encoding="utf-8") as f:
    mapping = json.load(f)

map_df = pd.DataFrame(mapping)

# sanity: keep only needed columns & drop duplicates
map_df = map_df[["model_id", "processed", "original"]].dropna()
map_df = map_df.drop_duplicates(subset=["model_id", "processed"], keep="first")

# 3) figure out which column in df contains the processed review text
# change this to your actual column name in the csv:
PROCESSED_COL = "reviews"  # <-- change if needed (e.g. "text", "content", etc.)

if PROCESSED_COL not in df.columns:
    raise ValueError(f"Couldn't find '{PROCESSED_COL}' in CSV columns: {list(df.columns)}")

# 4) merge
out = df.merge(
    map_df,
    how="left",
    left_on=["model_id", PROCESSED_COL],
    right_on=["model_id", "processed"],
)

# optional: drop the mapping key col from the right side
out = out.drop(columns=["processed"])

# 5) report match rate
missing = out["original"].isna().sum()
print(f"Rows: {len(out)} | Missing originals: {missing}")

# 6) save
# out.to_csv(OUT_CSV, index=False)
print(f"Wrote: {OUT_CSV}")

# ----------- ORIGINAL DATASET STATS -----------

print("\n===== ORIGINAL DATASET =====")

# total reviews
total_reviews_original = len(df)
print("Total reviews:", total_reviews_original)

# category distribution
category_counts_original = df["mode_2"].value_counts().sort_index()
print("\nCategory distribution (counts):")
print(category_counts_original)

print("\nCategory distribution (percent):")
print((category_counts_original / total_reviews_original * 100).round(2))

# unique models per label
models_per_label = df.groupby("mode_2")["model_id"].nunique()
print("\nUnique models per label:")
print(models_per_label)

# total unique models
total_models = df["model_id"].nunique()
print("\nTotal unique models:", total_models)


# ----------- SAMPLED DATASET STATS -----------

print("\n===== SAMPLED DATASET =====")

# total reviews
total_reviews_sample = len(sampled_df)
print("Total reviews:", total_reviews_sample)

# category distribution
category_counts_sample = sampled_df["mode_2"].value_counts().sort_index()
print("\nCategory distribution (counts):")
print(category_counts_sample)

print("\nCategory distribution (percent):")
print((category_counts_sample / total_reviews_sample * 100).round(2))

# unique models per label in sample
models_per_label_sample = sampled_df.groupby("mode_2")["model_id"].nunique()
print("\nUnique models per label (sample):")
print(models_per_label_sample)

# total unique models in sample
total_models_sample = sampled_df["model_id"].nunique()
print("\nTotal unique models in sample:", total_models_sample)


comparison = pd.DataFrame({
    "original": category_counts_original,
    "sample": category_counts_sample
})

comparison["original_%"] = (comparison["original"] / total_reviews_original * 100).round(2)
comparison["sample_%"] = (comparison["sample"] / total_reviews_sample * 100).round(2)

print("\n===== DISTRIBUTION COMPARISON =====")
print(comparison)
