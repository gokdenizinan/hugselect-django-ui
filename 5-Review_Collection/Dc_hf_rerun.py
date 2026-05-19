import json

with open("1-MODEL_FILTERING/N_model_dict.json", "r", encoding = "utf-8") as f:
    model_dict = json.load(f)

with open("5-REVIEW_COLLECTION/united_reviews/hf_reviews_united.json", "r", encoding="utf-8") as f:
    hf_dict = json.load(f)

total_topics = [entry["model_id"] for entry in model_dict.values()]

hf_topics = [entry["model_id"] for entry in hf_dict]

only_total = list(set(total_topics) - set(hf_topics))
only_hf = list(set(hf_topics) - set(total_topics))

print(len( only_hf))
print(len( only_total))

filename = f"5-REVIEW_COLLECTION/hf_run_again.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(only_total, f, indent=2, ensure_ascii=False)