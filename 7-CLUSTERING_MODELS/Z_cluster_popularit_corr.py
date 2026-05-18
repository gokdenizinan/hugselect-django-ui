import pandas as pd
import json

# Load CSV
df = pd.read_csv("7-CLUSTERING_MODELS/clusters_improved_2/family_assignments_organized.csv")

# Load JSON (model_id -> likes)
with open("1-MODEL_FILTERING/N_sorted_model_likes_P9.json", "r") as f:
    likes_dict = json.load(f)

# Filter rows where family_root is "Other / Unclear"
filtered_df = df[df["family_root"] == "Other / Unclear"]

# Map likes into the dataframe
filtered_df["likes"] = filtered_df["model_id"].map(likes_dict)

# Drop models that don't have a like count (optional but recommended)
filtered_df = filtered_df.dropna(subset=["likes"])

# Sort by likes descending
top_10 = filtered_df.sort_values(by="likes", ascending=False).head(100)

# Print results
print(top_10[["model_id", "likes"]])
for i in top_10["model_id"]:
    print(i)