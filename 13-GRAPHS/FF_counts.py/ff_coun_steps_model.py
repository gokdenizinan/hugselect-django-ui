import json

# with open("2-NP_EXTRACTION/NP_X5.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

with open("4-LLM_FEATURE_ORGANIZATION/NP_GG_fin.json", "r", encoding="utf-8") as f:
    data = json.load(f)

unique_models = set()

for entry in data.values():
    model_ids = entry.get("model_id", [])
    
    if isinstance(model_ids, list):
        for model in model_ids:
            if isinstance(model, str):
                unique_models.add(model)

print("Number of unique models:", len(unique_models))