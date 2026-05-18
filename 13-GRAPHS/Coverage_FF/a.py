
import json

# Load the JSON dictionary from a file
with open("4-LLM_FEATURE_ORGANIZATION/NP_GG_fin.json", "r", encoding="utf-8") as f:
    data = json.load(f)

filtered_noun_phrases = [
    {
        "noun_phrase": item["noun_phrase"],
        "count": item["count"]
    }
    for item in data.values()
    if 50 <= item.get("count", 0) <= 300
]

print(filtered_noun_phrases)
print(len(filtered_noun_phrases))