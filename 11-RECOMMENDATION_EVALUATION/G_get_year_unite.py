import json

with open("11-RECOMMENDATION_EVALUATION/model_to_year.json") as f1, open("11-RECOMMENDATION_EVALUATION/model_to_year_more.json") as f2:
    dict1 = json.load(f1)
    dict2 = json.load(f2)

merged = dict1 | dict2  # merge

with open("11-RECOMMENDATION_EVALUATION/model_to_year_united.json", "w") as f:
    json.dump(merged, f, indent=4)