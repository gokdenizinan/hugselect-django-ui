import json

with open("1-MODEL_FILTERING/N_model_likes_10k_P4.json", "r", encoding = "utf-8") as f:
    p4 = json.load(f)

with open("1-MODEL_FILTERING/N_model_likes_10k_y1.json", "r", encoding = "utf-8") as f:
    y1 = json.load(f)


with open("1-MODEL_FILTERING/N_duplicate_model_names.json", "r", encoding = "utf-8") as f:
    y1_dup = json.load(f)

p4_list = list(p4.keys())[:300]
y1_list = list(y1.keys())[:5000]


p4_set = set(p4_list)
y1_set = set(y1_list)
y1_dup_set = set(y1_dup)

common_models = p4_set.intersection(y1_set)
print(len(p4_set))
print(f"Common models in both sets: {len(common_models)}")
# print(y1_dup_set - common_models)