import json
import os
from collections import defaultdict
import re

# === LOAD DATA ===
with open("1-MODEL_FILTERING/N_model_dict.json", "r", encoding="utf-8") as f:
    model_dict = json.load(f)

# print("1")
# with open("5-REVIEW_COLLECTION/united_reviews/stackoverflow_united.json", "r", encoding="utf-8") as f:
#     stack_dict = json.load(f)
# print("2")

# with open("5-REVIEW_COLLECTION/united_reviews/hf_reviews_united.json", "r", encoding="utf-8") as f:
#     hf_dict = json.load(f)
# print("3")

# with open("5-REVIEW_COLLECTION/united_reviews/reddit_united.json", "r", encoding="utf-8") as f:
#     reddit_dict = json.load(f)
# print("4")

# === OUTPUT FOLDER ===
# output_folder = "5-REVIEW_COLLECTION/united_reviews_by_model"
# os.makedirs(output_folder, exist_ok=True)


print("1")
with open("5-REVIEW_COLLECTION/stackoverflow_already_united/stackoverflow_3.json", "r", encoding="utf-8") as f:
    stack_dict = json.load(f)
print("2")

with open("5-REVIEW_COLLECTION/reddit_already_united/reddit_posts_26.json", "r", encoding="utf-8") as f:
    reddit_dict = json.load(f)
print("4")

output_folder = "5-REVIEW_COLLECTION/united_run_again"
os.makedirs(output_folder, exist_ok=True)


# with open("5-REVIEW_COLLECTION/united_reviews/reddit_united.json", "r", encoding="utf-8") as f:
#     reddit_dict = json.load(f)
# print("4")

# === HELPER FUNCTIONS ===
def get_model_ids_by_name(models_dict, model_name):
    """Return all model_ids for a given model_name."""
    return [
        info["model_id"]
        for info in models_dict.values()
        if info.get("model_name") == model_name
    ]


def ensure_model_entry(united_reviews, model_id, topic_name):
    """Ensure a model entry exists; if not, create it."""
    if model_id not in united_reviews:
        united_reviews[model_id] = {
            "model_id": model_id,
            "topic": topic_name,
            "reddit": [],
            "hf": [],
            "stack": []
        }
    return united_reviews[model_id]


def add_reddit_reviews(united_reviews, reddit_reviews):
    for reddit_topic in reddit_reviews:
        topic_name = reddit_topic.get("topic")
        model_ids = get_model_ids_by_name(model_dict, topic_name)
        if not model_ids:  # fallback if not found
            model_ids = [topic_name]

        for model_id in model_ids:
            topic_entry = ensure_model_entry(united_reviews, model_id, topic_name)

            for post in reddit_topic.get("posts", []):
                comms = post.get("comments", [])
                le_comms = 0
                if comms:
                    le_comms = len(comms) + 2
                reddit_entry = {
                    "title": post.get("title"),
                    "score": post.get("score"),
                    "subreddit": post.get("subreddit"),
                    "body": post.get("body"),
                    "comments": comms,
                    "num_rev": le_comms
                }
                topic_entry["reddit"].append(reddit_entry)


def add_hf_reviews(united_reviews, hf_reviews):
    for hf_post in hf_reviews:
        model_id = hf_post.get("model_id", "")
        topic_name = model_id.split("/")[-1] if "/" in model_id else model_id
        topic_entry = ensure_model_entry(united_reviews, model_id, topic_name)
        comms = [c.get("body") for c in hf_post.get("comments", []) if "body" in c]
        le_comms = 0
        if comms:
            le_comms = len(comms) + 2
        hf_entry = {
            "title": hf_post.get("title"),
            "body": hf_post.get("body", ""),
            "comments": comms,
            "num_rev" : le_comms
        }

        topic_entry["hf"].append(hf_entry)


def add_stack_reviews(united_reviews, stack_reviews):
    for stack_post in stack_reviews:
        topic_name = stack_post.get("searched_topic", "")
        model_ids = get_model_ids_by_name(model_dict, topic_name)
        if not model_ids:
            model_ids = [topic_name]

        for model_id in model_ids:
            topic_entry = ensure_model_entry(united_reviews, model_id, topic_name)

            comments_list = [
                answer.get("body", "")
                for answer in stack_post.get("answers", [])
                if answer.get("body")
            ]
            le_comms = 0
            if comments_list:
                le_comms = len(comments_list) + 2
            
            stack_entry = {
                "title": stack_post.get("title"),
                "score": stack_post.get("score"),
                "body": stack_post.get("body", ""),
                "comments": comments_list,
                "num_rev" : le_comms
            }

            topic_entry["stack"].append(stack_entry)


# === BUILD UNIFIED DICTIONARY ===
united_reviews = {}
print("5")

add_reddit_reviews(united_reviews, reddit_dict)
print("6")

# add_hf_reviews(united_reviews, hf_dict)
# print("7")

add_stack_reviews(united_reviews, stack_dict)
print("8")



# === SAVE ONE FILE PER MODEL ===
for model_id, data in united_reviews.items():
    safe_name = re.sub(r"[^\w\-_\.]", "_", model_id)  # sanitize filename
    output_path = os.path.join(output_folder, f"{safe_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Created {len(united_reviews)} JSON files in '{output_folder}'")
