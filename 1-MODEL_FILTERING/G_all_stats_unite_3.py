
import json

with open("1-MODEL_FILTERING/hf_model_all_stats.json", "r", encoding="utf-8") as f:
    all_stats_0 = json.load(f)

with open("1-MODEL_FILTERING/hf_model_all_stats_different.json", "r", encoding="utf-8") as f:
    all_stats_1 = json.load(f)

with open("1-MODEL_FILTERING/hf_model_all_stats_different_2.json", "r", encoding="utf-8") as f:
    all_stats_2 = json.load(f)

with open("1-MODEL_FILTERING/hf_model_all_stats_different_2.json", "r", encoding="utf-8") as f:
    all_stats_3 = json.load(f)
#######3
with open("1-MODEL_FILTERING/hf_gone_models.json", "r", encoding="utf-8") as f:
    gone_0 = json.load(f)

with open("1-MODEL_FILTERING/hf_gone_models_different.json", "r", encoding="utf-8") as f:
    gone_1 = json.load(f)

with open("1-MODEL_FILTERING/hf_gone_models_different_2.json", "r", encoding="utf-8") as f:
    gone_2 = json.load(f)

with open("1-MODEL_FILTERING/hf_gone_models_different_3.json", "r", encoding="utf-8") as f:
    gone_3 = json.load(f)

with open("1-MODEL_FILTERING/N_model_dict.json", "r", encoding="utf-8") as f:
    model_dict = json.load(f)

print(len(model_dict))

print(len(all_stats_0))
print(len(all_stats_1))
print(len(all_stats_2))
merged_se = all_stats_0 + all_stats_1 + all_stats_2
topics_merge = [entry["model_id"] for entry in merged_se] #29 k suanki ru
print(len(topics_merge))
print(len(set(topics_merge)))

merge_gon = gone_0 + gone_1 + gone_2 + gone_3 
print(len(merge_gon))
print(len(set(merge_gon)))

print("###############")
topics = [entry["model_id"] for entry in model_dict.values()] #29 k suanki ru

new_all_stats = []
for mod in topics_merge: 
    if mod not in topics: 
        new_all_stats.append(mod)

print("THIS IS :", len(model_dict))
print("should be:", len(topics))
print("is", len(new_all_stats))

##############check run and dedup run and remove

with open("5-REVIEW_COLLECTION/N_model_dict_run.json", "r", encoding="utf-8") as f:
    N_model_dict_run = json.load(f)

with open("5-REVIEW_COLLECTION/N_model_dict_remove.json", "r", encoding="utf-8") as f:
    N_model_dict_remove = json.load(f)

with open("5-REVIEW_COLLECTION/model_dict_original.json", "r", encoding="utf-8") as f:
    original_dict = json.load(f)

print(len(model_dict))
print(len(original_dict))
print(len(N_model_dict_run))
print(len(N_model_dict_remove))
print(len(N_model_dict_run) + len(original_dict) - len(N_model_dict_remove))
