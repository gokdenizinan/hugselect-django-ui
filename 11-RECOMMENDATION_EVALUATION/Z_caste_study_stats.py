import json

with open("11-RECOMMENDATION_EVALUATION/OUTPUT_F.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    
for sample_id, record in data.items():
    print(record.get("title"))