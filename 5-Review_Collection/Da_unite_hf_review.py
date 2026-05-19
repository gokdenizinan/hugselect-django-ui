import json

# Load data
with open("5-REVIEW_COLLECTION/united_reviews/hf_reviews_united.json", "r", encoding="utf-8") as f:
    np_dict = json.load(f)

with open("5-REVIEW_COLLECTION/hf_reviews_13_check_40853.json", "r", encoding="utf-8") as f:
    np_dict_2 = json.load(f)

# Merge and deduplicate by question_id
united_dict = np_dict + np_dict_2

# Deduplicate by model_id + disc_num
seen = set()
dedup_reviews = []

for review in united_dict:
    key = (review["model_id"], review["disc_num"])
    if key not in seen:
        dedup_reviews.append(review)
        seen.add(key)


# Save united dictionary
with open("5-REVIEW_COLLECTION/united_reviews/hf_reviews_united.json", "w", encoding="utf-8") as f:
    json.dump(dedup_reviews, f, indent=2, ensure_ascii=False)

def normalize_topic(topic: str) -> str:
    return topic.strip().lower()

unique_topics_1 = set(normalize_topic(d["model_id"]) for d in np_dict)
unique_topics_2 = set(normalize_topic(d["model_id"]) for d in np_dict_2)
unique_topics_united = set(d["model_id"] for d in united_dict)

# Print stats
print("-----")
print("DICT 1:", len(np_dict))
print("Topics 1: ", len(unique_topics_1) )
print("DICT 2:", len(np_dict_2))
print("Topics 2: ", len(unique_topics_2 ))
print("DICT1+DICT2:", len(np_dict) + len(np_dict_2))
print("TOTAL:", len(united_dict))
print("Topics DICT1+DICT2: ", len(unique_topics_1) + len(unique_topics_2 ))
print("Topics Total: ", len(unique_topics_united) )

print("TOTAL DEDUP:", len(dedup_reviews))


print("-----")

# # Combine and deduplicate done topics
# # Find repeating topics (intersection of both sets)
# repeating_topics = unique_topics_1 & unique_topics_2

# # Print repeating topics
# print("Repeating Topics in both Dict 1 and Dict 2:")
# for topic in repeating_topics:
#     print(topic)
