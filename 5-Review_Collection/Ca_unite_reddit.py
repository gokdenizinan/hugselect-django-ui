import json

# Load data
with open("5-REVIEW_COLLECTION/united_reviews/reddit_united.json", "r", encoding="utf-8") as f:
    np_dict = json.load(f)

with open("5-REVIEW_COLLECTION/reddit_already_united/reddit_posts_24_check_14500.json", "r", encoding="utf-8") as f:
    np_dict_2 = json.load(f)

# 1. Combine the two lists
combined = np_dict + np_dict_2

# 3. Deduplicate by 'topic' (keep first occurrence, remove duplicates)
seen_topics = set()
deduped = []
for entry in combined:
    topic = entry.get("topic")
    if topic and topic not in seen_topics:
        deduped.append(entry)
        seen_topics.add(topic)

# Print stats
print("-----")
print("DICT 1:", len(np_dict))
print("DICT 2:", len(np_dict_2))
print("DICT1+DICT2:", len(np_dict) + len(np_dict_2))
print("COMBINED", len(combined))
print("DEDUPED:", len(deduped))
print("-----")

# 4. Save deduplicated list to a new JSON file
filename = "5-REVIEW_COLLECTION/united_reviews/reddit_united.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(deduped, f, indent=2, ensure_ascii=False)