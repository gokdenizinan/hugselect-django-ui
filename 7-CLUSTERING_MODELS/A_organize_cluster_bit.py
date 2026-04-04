import pandas as pd
import json

df = pd.read_csv("7-CLUSTERING_MODELS/clusters_improved/family_assignments.csv")
df2 = pd.read_csv("7-CLUSTERING_MODELS/level1_domain_assignments_improved.csv")
print(df.columns)
print(df2.columns)

# Count values for each column
# print(df["assigned_modality"].value_counts().head(10))
# print(df["family_root"].value_counts().head(10))
# print(df["family_child"].value_counts().head(10))


repeated_list = df["family_root"].value_counts()
repeated_list = repeated_list[repeated_list > 1].index.tolist()
repeated_list = repeated_list[:100]
with open("7-CLUSTERING_MODELS/clusters_improved/family_root_list.json", "w") as f:
    json.dump(repeated_list, f, indent=2)

cols =['model_id', 'model_name', 'assigned_modality', "task", 'family_root',
       'family_child', 'assignment_method', 'family_confidence',
       'candidate_root_raw', 'candidate_root_norm', 'size_variant',
       'tuning_variant', 'quantization_variant', 'domain_variant',
       'base_models', 'pipeline_tag', 'effective_pipeline_tag', 'library_name', 'model_type', 'tags',
        ]


# Keep only the columns you want
new_df = df[cols]

# Save to a new CSV
new_df.to_csv("7-CLUSTERING_MODELS/clusters_improved/family_assignments_organized.csv", index=False)

