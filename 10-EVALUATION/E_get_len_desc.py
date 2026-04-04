
import pandas as pd
import json
# Example: df already exists
# df = pd.read_csv(...)

df = pd.read_csv("10-EVALUATION/model_ffs_eval_B_with_row_accuracy.csv")


# 1. Count words in the Descriptions column
df["description_word_count"] = (
    df["Descriptions"]
    .astype(str)
    .str.split()
    .str.len()
)

# 2. Compute correlation
pearson_corr = df["description_word_count"].corr(df["#Features"], method="pearson")
spearman_corr = df["description_word_count"].corr(df["#Features"], method="spearman")

print("Pearson correlation:", pearson_corr)
print("Spearman correlation:", spearman_corr)

# CHECK ALREADY DONE TECHNICAL WITH MYLIST
import ast
import pandas as pd

def to_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if x in ("", "[]", "]", "["):
            return []
        try:
            v = ast.literal_eval(x)  # safely parses "['a','b']"
            return v if isinstance(v, list) else [v]
        except Exception:
            # fallback: treat as a single feature string
            return [x]
    return [x]

df["Features_parsed"] = df["Features"].apply(to_list)
features_set = set(df["Features_parsed"].explode().dropna())


with open("4-LLM_FEATURE_ORGANIZATION/NP_info_global_united.json", "r", encoding="utf-8") as f:
    my_dict = json.load(f)

noun_phrase_set = {
    v["noun_phrase"]
    for v in my_dict.values()
    if "noun_phrase" in v
}
overlap = features_set & noun_phrase_set
count = len(overlap)

print(len(features_set))
print("Number of matches:", count)

import json

with open("2-NP_EXTRACTION/NP_global_dictionary_comfy.json", "r", encoding="utf-8") as f:
    data = json.load(f)
# Sort by count (descending) and take top 100
top_100 = sorted(
    data.values(),
    key=lambda x: x["count"],
    reverse=True
) # [:100]

# Optionally keep only noun_phrase and count
top_100_clean = [
    {"noun_phrase": item["noun_phrase"], "count": item["count"]}
    for item in top_100
]

# Save to JSON file
with open("10-EVALUATION/testing_nps.json", "w", encoding="utf-8") as f:
    json.dump(top_100_clean, f, ensure_ascii=False, indent=2)




# DESCRIPTION LENGTH EXTRACTION SCRIPT FOR TOP_K FILTERING
# import os
# import json

# input_folder = "HF-Models-T6"
# output_file = "10-EVALUATION/modelid_to_description_length.json"

# modelid_to_len = {}

# for filename in os.listdir(input_folder):
#     if filename.endswith(".json"):
#         file_path = os.path.join(input_folder, filename)

#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         model_id = data.get("modelID")
#         description = data.get("description", "")

#         if model_id is not None:
#             modelid_to_len[model_id] = len(description)

# # save the resulting dictionary
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(modelid_to_len, f, indent=2)

# print(f"Saved {len(modelid_to_len)} entries to {output_file}")
