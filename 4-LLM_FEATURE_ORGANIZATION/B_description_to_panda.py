import pandas as pd
from collections import defaultdict
import json

# # Load processed info dictionary
# with open("4-LLM_FEATURE_ORGANIZATION/NP_info_global_suan_fin.json", "r", encoding="utf-8") as f:
#     np_info_1 = json.load(f)

# with open("4-LLM_FEATURE_ORGANIZATION/NP_info_global_united.json", "r", encoding="utf-8") as f:
#     np_info_2 = json.load(f)

# with open("2-NP_EXTRACTION/NP_global_dictionary_comfy.json", "r", encoding="utf-8") as f:
#     np_info = json.load(f) 

# with open("4-LLM_FEATURE_ORGANIZATION/NP_info_global_united_suan.json", "r", encoding="utf-8") as f:
#     np_info = json.load(f)

with open("4-LLM_FEATURE_ORGANIZATION/NP_GG_fin.json", "r", encoding="utf-8") as f:
    np_info = json.load(f)

# with open("2-NP_EXTRACTION/NP_compfy_A.json", "r", encoding="utf-8") as f:
#     np_info = json.load(f)
# with open("4-LLM_FEATURE_ORGANIZATION/NP_O.json", "r", encoding="utf-8") as f:
#     np_info = json.load(f)
# Merge and clean up
# print(len(np_info_1))
# print(len(np_info_2))
# for entry in np_info_1.values():
#     model_ids = entry.get("model_id", [])
#     entry["author"] = [m.split("__", 1)[0] for m in model_ids]
#     entry["model_id"] = [m.replace("__", "/") for m in model_ids]

# for entry in np_info_2.values():
#     model_ids = entry.get("model_id", [])
#     entry["author"] = [m.split("/", 1)[0] for m in model_ids]

# np_info = np_info_1 | np_info_2  # Merge dictionaries


# output_path = "4-LLM_FEATURE_ORGANIZATION/NP_info_global_united_suan.json"

# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(np_info, f, indent=2, ensure_ascii=False)

# -------- SENTENCE-LEVEL SUMMARY --------
# Build rows for all NP entries
# rows = []
# for key, entry in np_info.items():
#     if "Description" in entry["info"] and "Technical" in entry["info"]:
#         rows.append({
#             "Feature": entry["noun_phrase"],
#             "Technical": entry["info"]["Technical"],
#             "Characteristic": entry["info"]["Characteristic"],
#             "Functionality": entry["info"]["Functionality"],
#             "Description": entry["info"]["Description"],
#             "Base Model": entry["base_author"],
#             "Base Author": entry["base_model"],
#             "Count": entry["count"],
#             "#Model_id": len(entry["model_id"]),
#             "#Authors": len(entry["author"]),
#             "Rep Sentences": entry["representative_sentences"],
#         })

# # Create DataFrame
# df_np_info = pd.DataFrame(rows)

# print("The number of Features with info:", df_np_info.shape[0])
# df_np_info.to_csv("4-LLM_FEATURE_Organization/df_feature_info_global_suan.csv", index=False, encoding="utf-8")


# -------- MODEL-LEVEL SUMMARY --------
import pandas as pd

# Assume `data` dictionary is already defined (with multiple NP entries)

# Step 1: Build a mapping from model_id to associated technical noun phrases
# Step 1: Build mapping model_id -> noun phrases and descriptions
model_to_nps = {}
model_to_descs = {}
import json
import os

folder_path = "HF-Models-T6"

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            model_id = data.get("modelID")
            description = data.get("description")

            if model_id is not None:
                model_to_descs[model_id] = description

for entry in np_info.values():
    # if entry.get("info", {}).get("Technical") == "yes": # DURUMA GORE
    for model_id in entry.get("model_id", []):
        model_to_nps.setdefault(model_id, []).append(entry.get("noun_phrase"))

# Step 2: Build DataFrame
rows = []
for model_id, np_list in model_to_nps.items():
    model_id = model_id.replace("__", "/") # DURUMA GORE
    rows.append({
        "ModelID": model_id,
        "Features": np_list,
        "#Features": len(np_list),
        "Descriptions": model_to_descs.get(model_id, []),
    })

print(len(model_to_descs))
df_model_info = pd.DataFrame(rows)

import ast

import ast
import pandas as pd

import ast
import pandas as pd
import numpy as np

def to_list(x):
    # NaN / None (scalar only)
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    # already list / tuple / numpy array
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)

    # string case
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return list(v)
            return [v]
        except (ValueError, SyntaxError):
            # fallback for malformed list strings
            s = s.strip("[]")
            return [
                p.strip(" '\"\t\n")
                for p in s.split(",")
                if p.strip(" '\"\t\n")
            ]

    # everything else
    return [x]


all_items = set(
    item
    for lst in df_model_info["Features"].map(to_list)
    for item in lst
    if pd.notna(item) and str(item).strip() != ""
)
print("Unique features in the sample", len(all_items))


print("The number of Models with info:", df_model_info.shape[0])
df_model_info.to_csv("10-EVALUATION/model_ffs_new.csv", index=False, encoding="utf-8")

