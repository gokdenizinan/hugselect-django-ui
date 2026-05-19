import json
from collections import defaultdict 
import re

with open("5-REVIEW_COLLECTION/model_dict_original.json", "r", encoding = "utf-8") as f:
    model_dict = json.load(f)

with open("5-REVIEW_COLLECTION/united_reviews/stackoverflow_united.json", "r", encoding = "utf-8") as f:
    stack_dict = json.load(f)

with open("5-REVIEW_COLLECTION/united_reviews/hf_reviews_united.json", "r", encoding = "utf-8") as f:
    hf_dict = json.load(f)

with open("5-REVIEW_COLLECTION/united_reviews//reddit_united.json", "r", encoding = "utf-8") as f:
    reddit_dict = json.load(f)

### STACK
stack_models = set()

for item in stack_dict:
    for key, value in item.items():
        stack_models.add(item["searched_topic"])

print("STACK:", len(stack_models))

### HF

hf_models = set()

for item in hf_dict:
    for key, value in item.items():
        hf_models.add(item["model_id"])

print("HF:", len(hf_models))

### REDDIT
reddit_models = set()

for item in reddit_dict:
    for key, value in item.items():
        reddit_models.add(item["topic"])

print("REDDIT:", len(reddit_models))

### KONTROL TOPIC IN REVIEW

topic_united = {
    "model_id": None,
    "topic": None,
    "stack": [{
        "title": None,
        "score": None,
        "body": None,
        "comments": []
    }],
    "hf": [{
        "title": None,
        "body": None,
        "comments": []
    }],
    "reddit": [{
        "title": None,
        "score": None,
        "subreddit": None,
        "body": None,
        "comments": []
    }]
}


def get_model_ids_by_name(models_dict, model_name):
    """
    Returns a list of model_ids corresponding to the given model_name.
    
    :param models_dict: Dictionary of dictionaries containing model info
    :param model_name: Name of the model to search for
    :return: List of model_ids matching the model_name
    """
    return [
        info["model_id"]
        for info in models_dict.values()
        if info.get("model_name") == model_name
    ]

result = get_model_ids_by_name(model_dict, "my-pet-cat-xzg")

def add_reddit_reviews(united_reviews, reddit_reviews):
    """
    Adds Reddit reviews to the supreme list `united_reviews`.
    
    Parameters:
    - united_reviews: list of topics in supreme format
    - reddit_reviews: list of Reddit review dictionaries
    """
    for reddit_topic in reddit_reviews:
        # Check if topic already exists in united_reviews
        topic_name = reddit_topic.get("topic")
        model_ids = get_model_ids_by_name(model_dict, topic_name)
        for model in model_ids:
            topic_entry = next((t for t in united_reviews if t["model_id"] == model), None)
            
            # If not, create a new topic entry
            if not topic_entry:
                topic_entry = {
                    "model_id": model,
                    "topic": topic_name, 
                    "reddit": [],
                    "hf": [],
                    "stack": []
                }
                united_reviews.append(topic_entry)
            
            # Add Reddit posts to the topic
            for post in reddit_topic.get("posts", []):
                reddit_entry = {
                    "title": post.get("title"),
                    "score": post.get("score"),
                    "subreddit": post.get("subreddit"),
                    "body": post.get("body"),
                    "comments": post.get("comments", [])
                }
                topic_entry["reddit"].append(reddit_entry)



def add_hf_reviews(united_reviews, hf_reviews):
    """
    Adds Hugging Face reviews to the supreme list `united_reviews`.
    
    Parameters:
    - united_reviews: list of topics in supreme format
    - hf_reviews: list of HF review dictionaries
    """
    for hf_post in hf_reviews:
        model_id = hf_post.get("model_id", "")
        topic_name = model_id.split("/")[-1] if "/" in model_id else model_id

        # Check if topic already exists in united_reviews
        topic_entry = next((t for t in united_reviews if t["model_id"] == model_id), None)
        
        # If not, create a new topic entry
        if not topic_entry:
            topic_entry = {
                "model_id": model_id,
                "topic": topic_name,
                "reddit": [],
                "hf": [],
                "stack": []
            }
            united_reviews.append(topic_entry)

        # Prepare HF entry
        hf_entry = {
            "title": hf_post.get("title"),
            "body": hf_post.get("body", ""),
            "comments": [c.get("body") for c in hf_post.get("comments", []) if "body" in c]
        }

        # Append to topic's HF list
        topic_entry["hf"].append(hf_entry)

def add_stack_reviews(united_reviews, stack_reviews):
    """
    Adds Stack Overflow reviews to the supreme list `united_reviews`.
    
    Parameters:
    - united_reviews: list of topics in supreme format
    - stack_reviews: list of Stack Overflow review dictionaries
    """
    for stack_post in stack_reviews:
        # Extract topic from searched_topic
        topic_name = stack_post.get("searched_topic", "")
        model_ids = get_model_ids_by_name(model_dict, topic_name)
        for model in model_ids:
            # Check if topic already exists in united_reviews
            topic_entry = next((t for t in united_reviews if t["model_id"] == model), None)

            # If not, create a new topic entry
            if not topic_entry:
                topic_entry = {
                    "topic": topic_name,
                    "model_id": model,
                    "reddit": [],
                    "hf": [],
                    "stack": []
                }
                united_reviews.append(topic_entry)

            comments_list = []
            # Add answers as comments
            for answer in stack_post.get("answers", []):
                answer_body = answer.get("body", "")
                if answer_body:
                    comments_list.append(answer_body)

            stack_entry = {
                "title": stack_post.get("title"),
                "score": stack_post.get("score"),
                "body": stack_post.get("body", "") ,
                "comments": comments_list
            }

            # Append to topic's Stack Overflow list
            topic_entry["stack"].append(stack_entry)


united_reviews = []
# Suppose reddit_reviews is your Reddit dictionary list
add_reddit_reviews(united_reviews, reddit_dict)
add_hf_reviews(united_reviews, hf_dict)
add_stack_reviews(united_reviews, stack_dict)

# united_reviews now has Reddit posts in the supreme format


with open("5-REVIEW_COLLECTION/united_reviews/united_reviews.json", "w", encoding = "utf-8") as f:
    json.dump(united_reviews, f, indent=2, ensure_ascii=False)