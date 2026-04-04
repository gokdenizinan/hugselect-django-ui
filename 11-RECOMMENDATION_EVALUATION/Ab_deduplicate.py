import json

with open("11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_1.json", "r", encoding="utf-8") as f:
    original = json.load(f)

with open("11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_2.json", "r", encoding="utf-8") as f:
    batch1 = json.load(f)

with open("11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_3.json", "r", encoding="utf-8") as f:
    batch2 = json.load(f)
with open("11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_4.json", "r", encoding="utf-8") as f:
    batch3 = json.load(f)
with open("11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_5.json", "r", encoding="utf-8") as f:
    batch4 = json.load(f)

from collections import Counter

def count_decisions(items):
    return Counter(item.get("decision") for item in items)

original.extend(batch1)
original.extend(batch2)
original.extend(batch3)
original.extend(batch4)

original_count = len(original)

seen_ids = set()
filtered_list = [
    item for item in original
    if not (item["s2_paper_id"] in seen_ids or seen_ids.add(item["s2_paper_id"]))
]
new_count = len(filtered_list)
removed_count = original_count - new_count

print(f"Original items: {original_count}")
print(f"Removed items:  {removed_count}")
print(f"Remaining:      {new_count}")
print("=== === ===")
org_counts = count_decisions(original)
after_counts = count_decisions(filtered_list)

print("\nORIGINAL:")
for k, v in org_counts.items():
    print(f"  {k}: {v}")

print("\nAFTER:")
for k, v in after_counts.items():
    print(f"  {k}: {v}")



with open("11-RECOMMENDATION_EVALUATION/MORE_PAPERS/Candidates_SS_Batch_all_Dedup.json", "w", encoding="utf-8") as f:
    json.dump(filtered_list, f, ensure_ascii=False, indent=2)