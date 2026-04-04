import json

with open("11-RECOMMENDATION_EVALUATION/paper_model/hf_case_study_candidates_openalex.json", "r", encoding="utf-8") as f:
    original = json.load(f)

with open("11-RECOMMENDATION_EVALUATION/paper_model_2/hf_case_study_candidates_openalex_3.json", "r", encoding="utf-8") as f:
    new = json.load(f)

from collections import Counter

def count_decisions(items):
    return Counter(item.get("decision") for item in items)


ids_to_remove = {item["openalex_id"] for item in original}

original_count = len(new)

filtered_list = [
    item for item in  new
    if item["openalex_id"] not in ids_to_remove
]

new_count = len(filtered_list)
removed_count = original_count - new_count

print(f"Original items: {original_count}")
print(f"Removed items:  {removed_count}")
print(f"Remaining:      {new_count}")
print("=== === ===")
before_counts = count_decisions(new)
org_counts = count_decisions(original)
after_counts = count_decisions(filtered_list)

print("BEFORE:")
for k, v in before_counts.items():
    print(f"  {k}: {v}")

print("\nORIGINAL:")
for k, v in org_counts.items():
    print(f"  {k}: {v}")

print("\nAFTER:")
for k, v in after_counts.items():
    print(f"  {k}: {v}")



with open("11-RECOMMENDATION_EVALUATION/paper_model_2/hf_case_study_candidates_openalex_3F.json", "w", encoding="utf-8") as f:
    json.dump(filtered_list, f, ensure_ascii=False, indent=2)