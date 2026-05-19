import json

# Load data
with open("5-REVIEW_COLLECTION/united_reviews/stackoverflow_united.json", "r", encoding="utf-8") as f:
    np_dict = json.load(f)

# with open("5-REVIEW_COLLECTION/stackoverflow_already_united/stackoverflow_94.json", "r", encoding="utf-8") as f:
#     np_dict_2 = json.load(f)

with open("5-REVIEW_COLLECTION/stackoverflow_already_united/stackoverflow_89_checkpoint_100.json", "r", encoding="utf-8") as f:
    np_dict_2 = json.load(f)

# Merge and deduplicate by question_id
unique_dict = {d['question_id']: d for d in (np_dict + np_dict_2)}
united_dict = list(unique_dict.values())

# Save united dictionary
with open("5-REVIEW_COLLECTION/united_reviews/stackoverflow_united.json", "w", encoding="utf-8") as f:
    json.dump(united_dict, f, indent=2, ensure_ascii=False)

def normalize_topic(topic: str) -> str:
    return topic.strip().lower()

unique_topics_1 = set(normalize_topic(d["searched_topic"]) for d in np_dict)
unique_topics_2 = set(normalize_topic(d["searched_topic"]) for d in np_dict_2)
unique_topics_united = set(d["searched_topic"] for d in united_dict)

# Print stats
print("-----")
print("DICT 1:", len(np_dict))
print("Topics 1: ", len(unique_topics_1) )
print("DICT 2:", len(np_dict_2))
print("Topics 2: ", len(unique_topics_2 ))
print("DICT1+DICT2:", len(np_dict) + len(np_dict_2))
print("TOTAL:", len(united_dict))
print("Topics Total: ", len(unique_topics_united) )


print("-----")

# Combine and deduplicate done topics
done_topics = list(set(unique_topics_1 | unique_topics_2))

# Find repeating topics (intersection of both sets)
repeating_topics = unique_topics_1 & unique_topics_2

# Print repeating topics
print("Repeating Topics in both Dict 1 and Dict 2:")
for topic in repeating_topics:
    print(topic)


# Save updated done topics
with open("5-REVIEW_COLLECTION/done_topics.json", "w", encoding="utf-8") as f:
    json.dump(done_topics, f, indent=2, ensure_ascii=False)

print("Updated done_topics saved. Total:", len(done_topics))
