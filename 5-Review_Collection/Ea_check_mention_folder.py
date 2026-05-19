import os
import json
import re
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm  # progress bar

nltk.download("punkt")

# INPUT_FOLDER = "5-REVIEW_COLLECTION/united_reviews_by_model"
# OUTPUT_FOLDER = "5-REVIEW_COLLECTION/united_mentioned_reviews"


INPUT_FOLDER = "5-REVIEW_COLLECTION/united_run_again"
OUTPUT_FOLDER = "5-REVIEW_COLLECTION/united_run_again_mention_2"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_author(model):
    model_id = model.get('model_id', "")
    parts = model_id.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], ""

def get_author_dict(models):
    author_dict = {}
    for item in models:
        author, topic = get_author(item)
        if not topic:
            continue
        author_dict.setdefault(topic, set()).add(author)
    return author_dict

def extract_mentions_with_authors(text, topic, model_author, possible_authors):
    sentences = sent_tokenize(text)
    mentions = []
    candidates = []
    
    for a in possible_authors.get(topic, []):
        candidates.extend([
            a,
            f"{topic}/{a}",
            f"{topic}-{a}",
            f"{topic} {a}"
        ])
    
    for s in sentences:
        if re.search(rf"\b{re.escape(topic)}\b", s, re.IGNORECASE):
            mentioned_authors = [a for a in candidates if a.lower() in s.lower()]
            if mentioned_authors:
                if any(model_author.lower() in ma.lower() for ma in mentioned_authors):
                    mentions.append(s)
            else:
                mentions.append(s)
    return mentions

def extract_topic_mentions(data, possible_authors):
    post_keys = ['reddit', 'stack', 'hf']
    filtered_models = []

    for model in data:
        topic = model.get('topic', "")
        if not topic:
            continue
        model_author, _ = get_author(model)
        keep_model = False

        for key in post_keys:
            if key not in model:
                continue

            filtered_posts = []
            for post in model[key]:
                mentioned = []
                contents = [post.get('title', ''), post.get('body', '')] + post.get('comments', [])

                for content in contents:
                    if content:
                        mentioned.extend(extract_mentions_with_authors(content, topic, model_author, possible_authors))

                if mentioned:
                    post['mentioned'] = mentioned
                    for field in ['title', 'body', 'comments']:
                        post.pop(field, None)
                    filtered_posts.append(post)

            model[key] = filtered_posts
            if filtered_posts:
                keep_model = True

        if keep_model:
            filtered_models.append(model)

    return filtered_models

# --- Main loop over all JSON files in the folder ---

def load_mentioned_from_folder(folder_path):
    mentioned = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        mentioned.extend(data)
                    else:
                        mentioned.append(data)
                except json.JSONDecodeError:
                    print(f"⚠️ Could not parse {filename}")
    return mentioned

print("1")
all_models = load_mentioned_from_folder(INPUT_FOLDER)

print("2")

possible_authors = get_author_dict(all_models)
print("3")

mentioned = extract_topic_mentions(all_models, possible_authors)

print("4")

# === SAVE ONE FILE PER MODEL ===
for model_id in mentioned:
    fil_name = model_id.get("model_id","")
    safe_name = re.sub(r"[^\w\-_\.]", "_", fil_name)  # sanitize filename
    output_path = os.path.join(OUTPUT_FOLDER, f"{safe_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model_id, f, indent=2, ensure_ascii=False)

print("✅ Done! All processed files saved in:", OUTPUT_FOLDER)
