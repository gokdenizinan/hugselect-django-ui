import json

with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/mentioned_reviews_1.json", "r", encoding = "utf-8") as f:
    original = json.load(f)

#change according to need
with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/mentioned_reviews_new_1.json", "r", encoding = "utf-8") as f:
    new = json.load(f)

def get_model_names(mentioned):
    model_names = set()
    for model in mentioned: 
        model_names.add(model["topic"])
    return list(model_names)

def filter_new_mentions(new, original):
    original_names = get_model_names(original)
    new_names = get_model_names(new)

    for name in new_names:
        if name in original_names:
            print(f"old model found: {name}")
            new_names.remove(name)
    return new_names