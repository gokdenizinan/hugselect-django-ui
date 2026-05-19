import json

with open("5-REVIEW_COLLECTION/model_dict_original.json", "r", encoding = "utf-8") as f:
    model_dict = json.load(f)

with open("5-REVIEW_COLLECTION/N_model_dict_run.json", "r", encoding = "utf-8") as f:
    model_dict_rerun = json.load(f)

with open("5-REVIEW_COLLECTION/united_reviews/hf_reviews_united.json", "r", encoding = "utf-8") as f:
    hf_reviews = json.load(f)


def get_model_list(model_dict):
    model_ids = []
    for item in model_dict:
        model_ids.append(item["model_id"])
    return model_ids

def get_model_dict(model_dict):
    model_ids = []
    for key, value in model_dict.items():
        model_ids.append(value["model_id"])
    return model_ids

org_topics = get_model_dict(model_dict)
rerun_topics = get_model_list(model_dict_rerun)
hf_topics = get_model_list(hf_reviews)

combined = org_topics + rerun_topics

unique_to_list1 = [item for item in combined if item not in hf_topics]
unique_to_list2 = [item for item in hf_topics if item not in combined]

print("Unique to list1:", len(unique_to_list1))  # [1, 2]
print("Unique to list2:", len(unique_to_list2) ) # [5, 6]

filename = f"5-REVIEW_COLLECTION/hf_rerun_fin.json"
with open(filename, "w", encoding="utf-8") as f:
        json.dump(unique_to_list1, f, indent=2, ensure_ascii=False)