import json
with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_output_AB50_all_expanded.json", "r") as f:
    data = json.load(f)

from collections import defaultdict


total_reviews = len(data)
models = set()

category_review_counts = defaultdict(int)
category_models = defaultdict(set)

for item in data.values():
    category = item.get("Primary_Category", "Missing")
    model = item.get("model_id")

    if model:
        models.add(model)
        category_models[category].add(model)

    category_review_counts[category] += 1

total_models = len(models)

print("Total reviews:", total_reviews)
print("Total unique models:", total_models)
print()

for cat in category_review_counts:
    review_count = category_review_counts[cat]
    review_pct = review_count / total_reviews * 100

    model_count = len(category_models[cat])
    model_pct = model_count / total_models * 100 if total_models else 0

    print(f"{cat}")
    print(f"  Reviews: {review_count} ({review_pct:.2f}%)")
    print(f"  Models:  {model_count} ({model_pct:.2f}%)")

print("")
print("")

print("")
print("")

import pandas as pd

# Load CSV
# df = pd.read_csv("12-EVALUATION_QUAL/attribute_sample_2.csv")

df = pd.read_csv("12-EVALUATION_QUAL/quality_sample_A0.csv")
# Count occurrences of each label
counts = df["Primary_Category"].value_counts()

# Calculate percentage distribution
percentages = df["Primary_Category"].value_counts(normalize=True) * 100

# Combine into one table
distribution = pd.DataFrame({
    "Count": counts,
    "Percentage (%)": percentages
})

print(distribution)