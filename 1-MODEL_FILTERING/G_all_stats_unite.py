
import json

with open("1-MODEL_FILTERING/hf_model_all_stats.json", "r", encoding="utf-8") as f:
    all_stats_0 = json.load(f)

with open("1-MODEL_FILTERING/hf_model_all_stats_different.json", "r", encoding="utf-8") as f:
    all_stats_1 = json.load(f)

with open("1-MODEL_FILTERING/hf_model_all_stats_different_2.json", "r", encoding="utf-8") as f:
    all_stats_2 = json.load(f)

with open("1-MODEL_FILTERING/N_model_dict.json", "r", encoding="utf-8") as f:
    model_dict = json.load(f)

print(len(all_stats_1))
print(len(all_stats_2))
merged_se = all_stats_0 + all_stats_1 + all_stats_2
merged_set = set(merged_se)
merged = list(merged_set)
print(len(merged))

topics = [entry["model_id"] for entry in model_dict.values()] #29 k suanki ru

new_all_stats = []
for mod in merged: 
    mod_nam = mod["model_id"]
    if mod_nam in topics: 
        new_all_stats.append(mod)

print("should be:", len(topics))
print("is", len(new_all_stats))
