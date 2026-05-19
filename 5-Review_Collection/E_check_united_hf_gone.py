import json
import re
import matplotlib.pyplot as plt

with open("5-REVIEW_COLLECTION/united_reviews/united_reviews.json", "r", encoding = "utf-8") as f:
    united = json.load(f)

with open("5-REVIEW_COLLECTION/hf_gones/hf_0_gone_models.json", "r", encoding = "utf-8") as f:
    gone = json.load(f)

gone_in_united = []
for item in united:
    model_id = item["model_id"]
    if model_id in gone:
        gone_in_united.append(model_id)

for i in gone_in_united:
    print(i)

print("====")
print("total", len(gone_in_united))