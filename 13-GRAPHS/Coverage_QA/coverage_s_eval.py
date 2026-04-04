import pandas as pd
import numpy as np
import json

# ----------------------------
# 1) LOAD ORIGINAL DATASET
# ----------------------------
ORIG_CSV = "6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/sentiment_for_f5_united_3.csv"
orig_df = pd.read_csv(ORIG_CSV)

# ----------------------------
# 2) CREATE SAMPLE
# ----------------------------
total_sample_size = 500
min_per_class = 100

counts = orig_df["mode_2"].value_counts()

sample_sizes = {label: min_per_class for label in counts.index}
remaining = total_sample_size - sum(sample_sizes.values())

proportions = counts / counts.sum()
additional = (proportions * remaining).round().astype(int)

for label in sample_sizes:
    sample_sizes[label] += additional[label]

diff = total_sample_size - sum(sample_sizes.values())
if diff != 0:
    largest_class = counts.idxmax()
    sample_sizes[largest_class] += diff

print("Sample allocation:", sample_sizes)

sampled_df = (
    orig_df.groupby("mode_2", group_keys=False)
    .apply(lambda x: x.sample(n=sample_sizes[x.name], random_state=42).assign(mode_2=x.name))
    .reset_index(drop=True)
)

sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------------------
# 3) ORIGINAL STATS
# ----------------------------
total_reviews_original = len(orig_df)
category_counts_original = orig_df["mode_2"].value_counts().sort_index()
category_pct_original = (category_counts_original / total_reviews_original * 100).round(1)
models_per_label_original = orig_df.groupby("mode_2")["model_id"].nunique().sort_index()
total_models_original = orig_df["model_id"].nunique()

# ----------------------------
# 4) SAMPLE STATS
# ----------------------------
total_reviews_sample = len(sampled_df)
category_counts_sample = sampled_df["mode_2"].value_counts().sort_index()
category_pct_sample = (category_counts_sample / total_reviews_sample * 100).round(1)
models_per_label_sample = sampled_df.groupby("mode_2")["model_id"].nunique().sort_index()
total_models_sample = sampled_df["model_id"].nunique()

# ----------------------------
# 5) COMPARISON TABLE
# ----------------------------
table = pd.DataFrame({
    "Reviews (Original)": category_counts_original,
    "% (Original)": category_pct_original,
    "Unique Models (Original)": models_per_label_original,
    "Reviews (Sample)": category_counts_sample,
    "% (Sample)": category_pct_sample,
    "Unique Models (Sample)": models_per_label_sample,
})

table.loc["Total"] = [
    total_reviews_original,
    100.0,
    total_models_original,
    total_reviews_sample,
    100.0,
    total_models_sample,
]

print("\n===== DISTRIBUTION TABLE =====")
print(table)