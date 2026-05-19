# import json

# with open("5-REVIEW_COLLECTION/stackoverflow_united.json", "r", encoding="utf-8") as f:
#     STACK = json.load(f)

# with open("5-REVIEW_COLLECTION/model_dict_original.json", "r", encoding="utf-8") as f:
#     model_dict = json.load(f)

# with open("5-REVIEW_COLLECTION/done_topics.json", "r", encoding="utf-8") as f:
#     DONE = json.load(f)


# #------- SPLIT 1
# TOPICS = [entry["model_name"] for entry in model_dict.values()][0:29000]

# all_topics = {q["searched_topic"] for q in STACK}

# # Find which topics match (count each only once)
# matching_topics = all_topics.intersection(TOPICS)

# print("LEN MATCHIN TOPICS (:25K):" , len(matching_topics))


# #------- SPLIT 1 - DONE

# all_topics = {q["searched_topic"] for q in STACK}

# # Find which topics match (count each only once)
# matching_topics = all_topics.intersection(DONE)

# print("LEN MATCHIN TOPICS with done:" , len(matching_topics))

# #------- SPLIT 1 - DONW WITH TOPICS
# TOPICS = [entry["model_name"] for entry in model_dict.values()][0:29000]


# # Find which topics match (count each only once)
# matches = list(set(TOPICS) & set(DONE))

# print("LEN MATCHIN TOPICS AND DONE (:25K):" , len(matches))

# #------- SPLIT 1 - DONW WITH TOPICS
# TOPICS = [entry["model_name"] for entry in model_dict.values()]


# # Find which topics match (count each only once)
# matches = list(set(TOPICS) & set(DONE))

# print("LEN MATCHIN TOPICS AND DONE (FULL):" , len(matches))

# #------------- SPLIT 1 - 4000
# TOPICS = [entry["model_name"] for entry in model_dict.values()][4000:29000]

# all_topics = {q["searched_topic"] for q in STACK}

# # Find which topics match (count each only once)
# matching_topics = all_topics.intersection(TOPICS)

# print("LEN MATCHIN TOPICS (4K:25K):" , len(matching_topics))

# #-------- ALL
# TOPICS = [entry["model_name"] for entry in model_dict.values()]

# all_topics = {q["searched_topic"] for q in STACK}

# # Find which topics match (count each only once)
# matching_topics = all_topics.intersection(TOPICS)
# non_matching_topics = [t for t in TOPICS if t not in all_topics]

# print("LEN MATCHIN TOPICS(FULL):" , len(matching_topics))
# print("LEN NON-MATCHIN TOPICS (:25K):" , len(non_matching_topics))

# print("====")

# print("len stack:", len(STACK))
# print("len model_dict:", len(TOPICS))


# # with open("5-REVIEW_COLLECTION/llm_check_reviews/stack_check.json", "w", encoding="utf-8") as f:
# #     json.dump(idk, f, indent=2, ensure_ascii=False)


import json

with open("5-REVIEW_COLLECTION/united_reviews/stackoverflow_united.json", "r", encoding="utf-8") as f:
    stack_dict = json.load(f)

with open("5-REVIEW_COLLECTION/model_dict_original.json", "r", encoding="utf-8") as f:
    model_dict = json.load(f)

TOPICS = [entry["model_name"] for entry in model_dict.values()]

STACK = {q["searched_topic"] for q in stack_dict}

indexes_dict = {}

for item in STACK:
    indexes = [i for i, topic in enumerate(TOPICS) if topic == item]
    indexes_dict[item] = indexes

keys_over_33k = [key for key, values in indexes_dict.items() if all(v > 33000 for v in values)]

print(keys_over_33k)

with open("5-REVIEW_COLLECTION/llm_check_reviews/stack_check.json", "w", encoding="utf-8") as f:
    json.dump(indexes_dict, f, indent=2, ensure_ascii=False)