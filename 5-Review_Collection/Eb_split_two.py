import json

with open("5-REVIEW_COLLECTION/united_reviews/united_reviews.json", "r", encoding="utf-8") as f:
    mentioned = json.load(f)

with open("5-REVIEW_COLLECTION/model_dict.json", "r", encoding="utf-8") as f:
    model_dict = json.load(f)


def get_model_names(mentioned):
    model_names = set()
    for model in mentioned:
        model_names.add(model["topic"])
    return list(model_names)


names = get_model_names(mentioned)

all_models = [entry["model_name"] for entry in model_dict.values()]
line = all_models[4000:33000]
line_2 = all_models[:4000] + all_models[33000:]


common_items = [n for n in names if n in line]
non_common_items = [n for n in names if n in line_2]

common_set = set(common_items)
non_common_set = set(non_common_items)

# Option A: use remove (your original style, fixed with copies)
split_1 = mentioned[:]  # copy
for item in mentioned[:]:
    if item["topic"] in common_items:
        split_1.remove(item)

split_2 = mentioned[:]  # copy
for item in mentioned[:]:
    if item["topic"] in non_common_items:
        split_2.remove(item)

# Option B: much cleaner with list comprehension
split_1 = [item for item in mentioned if item["topic"] in common_items]
split_2 = [item for item in mentioned if item["topic"] in non_common_items]

print("SPLIT 1", len(split_1))
print("SPLIT 2", len(split_2))

with open("5-REVIEW_COLLECTION/llm_check_reviews/split_1.json", "w", encoding="utf-8") as f:
    json.dump(split_1, f, indent=2, ensure_ascii=False)

with open("5-REVIEW_COLLECTION/llm_check_reviews/split_2.json", "w", encoding="utf-8") as f:
    json.dump(split_2, f, indent=2, ensure_ascii=False)
