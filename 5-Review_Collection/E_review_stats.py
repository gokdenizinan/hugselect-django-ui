import json
import pandas as pd

with open("5-REVIEW_COLLECTION/model_dict_dedup.json", "r", encoding = "utf-8") as f:
    model_dict = json.load(f)

with open("5-REVIEW_COLLECTION/united_reviews/stackoverflow_united.json", "r", encoding = "utf-8") as f:
    stack_dict = json.load(f)

TOPICS = [entry["model_name"] for entry in model_dict.values()]#29 k suanki run
icindekiler = []
disindakiler = []
stack_topics = set()
for item in stack_dict:
    stack_topics.add(item["searched_topic"])
for top in TOPICS:
    if top in stack_topics:
        icindekiler.append(top)
    else:
        disindakiler.append(top)

print("TOPICS IN STACK:", len(icindekiler))
print("TOPICS NOT IN STACK:", len(disindakiler))

with open("5-REVIEW_COLLECTION/stack_disindakiler.json", "w", encoding = "utf-8") as f:
    json.dump(disindakiler, f, indent=2, ensure_ascii=False)