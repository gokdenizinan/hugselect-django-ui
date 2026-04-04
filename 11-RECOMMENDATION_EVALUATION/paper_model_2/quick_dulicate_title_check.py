import json

with open("11-RECOMMENDATION_EVALUATION/paper_model_2/matched_papers_summary.json", "r") as f:
    data = json.load(f)

from collections import Counter

titles = [d["title"] for d in data]
repeating_titles = [t for t, count in Counter(titles).items() if count > 1]

print(repeating_titles)
