import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---- Load CSV ----
df = pd.read_csv("7-CLUSTERING_MODELS/clusters_improved/organizedd/family_assignments_organized_again.csv")

exclude_unclear = True  # 🔁 toggle this

# ---- Load CSV ----

# ---- Drop rows with missing model_type ----
df = df.dropna(subset=["model_type"])

# ---- Optional: remove 'Other / Unclear' ----
if exclude_unclear:
    df = df[df["family_root"] != "Other / Unclear"]

# ---- Normalization function ----
def normalize(value):
    if not isinstance(value, str):
        return value
    
    value = value.lower().strip()
    value = re.sub(r'\d+$', '', value)   # remove trailing numbers
    value = re.sub(r'-+$', '', value)    # remove trailing hyphens
    
    return value

# ---- Apply normalization ----
df["family_root_norm"] = df["family_root"].apply(normalize)
df["model_type_norm"] = df["model_type"].apply(normalize)

# ---- Ground truth and predictions ----
y_true = df["family_root_norm"]
y_pred = df["model_type_norm"]

# ---- Metrics ----
accuracy = accuracy_score(y_true, y_pred)

precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

# ---- Output ----
print(f"Rows used: {len(df)} (exclude_unclear={exclude_unclear})")
print(f"Accuracy: {accuracy:.4f}")

print("\nMacro avg:")
print(f"  Precision: {precision_macro:.4f}")
print(f"  Recall:    {recall_macro:.4f}")
print(f"  F1-score:  {f1_macro:.4f}")

print("\nWeighted avg:")
print(f"  Precision: {precision_weighted:.4f}")
print(f"  Recall:    {recall_weighted:.4f}")
print(f"  F1-score:  {f1_weighted:.4f}")

print("=. = = == = = == ")

# ---- Extra stats ----

# Use original dataframe (before dropna/filtering)
df_full = pd.read_csv("7-CLUSTERING_MODELS/clusters_improved/organizedd/family_assignments_organized_again.csv")
# Normalize helper for consistent comparison
def is_unclear(val):
    if not isinstance(val, str):
        return False
    return val.strip().lower() == "other / unclear"

# 1. Rows where model_type is missing BUT family_root is NOT "Other / Unclear"
mask_model_missing = df_full["model_type"].isna()
mask_family_not_unclear = ~df_full["family_root"].apply(is_unclear)

count_missing_model_but_clear_family = (mask_model_missing & mask_family_not_unclear).sum()

# 2. Rows where family_root IS "Other / Unclear" BUT model_type HAS a value
mask_family_unclear = df_full["family_root"].apply(is_unclear)
mask_model_present = df_full["model_type"].notna()

count_unclear_family_but_model_present = (mask_family_unclear & mask_model_present).sum()

count_unclear_family_and_model_missing = (mask_family_unclear & df_full["model_type"].isna()).sum()

# ---- Print results ----
print("\nExtra stats:")
print(f"Missing model_type but clear family_root: {count_missing_model_but_clear_family}")
print(f"Unclear family_root but model_type present: {count_unclear_family_but_model_present}")
print(f"Unclear family_root and missing model_type: {count_unclear_family_and_model_missing}")


# from collections import Counter
# import json

# # ---- Normalize (reuse same logic for consistency) ----
# def normalize(value):
#     if not isinstance(value, str):
#         return value
    
#     value = value.lower().strip()
#     value = re.sub(r'\d+$', '', value)
#     value = re.sub(r'-+$', '', value)
    
#     return value

# # ---- Count values ----
# family_root_norm = df_full.loc[df_full["family_root"] != "Other / Unclear", "family_root"]
family_root_norm = df_full["family_root"].dropna().apply(normalize)
print("family root", len(family_root_norm))

# family_root_counter = Counter(family_root_norm)

# # ---- Save to JSON ----
# with open("7-CLUSTERING_MODELS/family_root_counts.json", "w", encoding="utf-8") as f:
#     json.dump(family_root_counter, f, indent=2)

family_child = df_full["family_child"].dropna()
print("family child", len(family_child))

# family_child_counter = Counter(family_child)
# with open("7-CLUSTERING_MODELS/family_child_counts.json", "w", encoding="utf-8") as f:
#     json.dump(family_child_counter, f, indent=2)

assigned_modality = df_full.loc[df_full["assigned_modality"] != "Other / Unclear", "assigned_modality"]
# assigned_modality = df_full["assigned_modality"].dropna()
print("modality", len(assigned_modality))

# assigned_modality_counter = Counter(assigned_modality)
# with open("7-CLUSTERING_MODELS/assigned_modality_counts.json", "w", encoding="utf-8") as f:
#     json.dump(assigned_modality_counter, f, indent=2)


pipeline_tag = df_full["pipeline_tag"].dropna()
print("pipeline tag", len(pipeline_tag))

# pipeline_tag_counter = Counter(pipeline_tag)
# with open("7-CLUSTERING_MODELS/pipeline_tag_counts.json", "w", encoding="utf-8") as f:
#     json.dump(pipeline_tag_counter, f, indent=2)

model_type = df_full["model_type"].dropna()
print("model type", len(model_type))

baseline_model = df_full["base_models"].dropna()
print("baseline model", len(baseline_model))
# print(f"\nSaved family_root_counts.json with {len(family_root_counter)} unique values")