import json
import re
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize
import spacy

# # Load SpaCy model
# nlp = spacy.load("en_core_web_sm")


# make sure punkt is downloaded (run once)
nltk.download("punkt")

# with open("5-REVIEW_COLLECTION/united_reviews/united_reviews.json", "r", encoding = "utf-8") as f:
#     united = json.load(f)

with open("5-REVIEW_COLLECTION/llm_check_reviews/split_1.json", "r", encoding = "utf-8") as f:
    united = json.load(f)

def get_author(model):
    model_id = model.get('model_id', "")
    author, topic = model_id.split("/")
    return author, topic

def get_author_dict(united):
    author_dict = {}
    for item in united:
        author, topic = get_author(item)
        if not topic:  # skip invalid entries
            continue
        if topic not in author_dict:
            author_dict[topic] = set()
        author_dict[topic].add(author)
    return author_dict

def extract_mentions_with_authors(text, topic, model_author, possible_authors):
    # sentences = sent_tokenize(text)
    # doc = nlp(str(text))
    sentences = sent_tokenize(text)
    mentions = []
    
    for s in sentences:
        if topic.lower() in s.lower():
            # Build candidate author mentions:
            candidates = []
            for a in possible_authors.get(topic, []):
                candidates.append(a)                          # author alone
                candidates.append(f"{topic}/{a}")            # topic/author
                candidates.append(f"{topic}-{a}")            # topic-author
                candidates.append(f"{topic} {a}")            # topic author (space)
            
            # Check if any candidate appears in sentence
            mentioned_authors = [a for a in candidates if a.lower() in s.lower()]
            
            if mentioned_authors:
                # Keep only if model's author matches
                if any(model_author.lower() in ma.lower() for ma in mentioned_authors):
                    mentions.append(s)
            else:
                # No author name mentioned → keep by default
                mentions.append(s)

    return mentions


def extract_topic_mentions(data, possible_authors):
    post_keys = ['reddit', 'stack', 'hf']
    filtered_models = []

    for model in data:
        topic = model['topic']
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


possible_authors = get_author_dict(united)
mentioned = extract_topic_mentions(united, possible_authors)

print("done")
with open("5-REVIEW_COLLECTION/united_reviews/mentioned_reviews_spacy_1.json", "w", encoding = "utf-8") as f:
    json.dump(mentioned, f, indent=2, ensure_ascii=False)


# ### PLOTTING
# with open("5-REVIEW_COLLECTION/united_reviews/mentioned_reviews.json", "r", encoding = "utf-8") as f:
#     mentioned = json.load(f)
    
# print(len(mentioned))

# with open("5-REVIEW_COLLECTION/united_reviews/united_reviews.json", "r", encoding = "utf-8") as f:
#     united = json.load(f)

# print(len(united))

# # Number of models before vs. after filtering
# plt.figure()
# plt.bar(["United (all)", "Mentioned (filtered)"], [len(united), len(mentioned)])
# plt.title("Number of Models Before vs After Filtering")
# plt.ylabel("Count of Models")
# plt.show()

# # Distribution of posts per model (before filtering)
# united_counts = [sum(len(model.get(src, [])) for src in ["reddit","stackoverflow","hf"]) for model in united]

# plt.figure()
# plt.hist(united_counts, bins=20)
# plt.title("Distribution of Post Counts per Model (United)")
# plt.xlabel("Number of Posts")
# plt.ylabel("Number of Models")
# plt.show()

# # Distribution of posts per model (after filtering)

# mentioned_counts = [sum(len(model.get(src, [])) for src in ["reddit","stackoverflow","hf"]) for model in mentioned]

# plt.figure()
# plt.hist(mentioned_counts, bins=20)
# plt.title("Distribution of Post Counts per Model (Mentioned)")
# plt.xlabel("Number of Posts")
# plt.ylabel("Number of Models")
# plt.show()

# # Top 10 models by number of filtered mentions

# model_mentions = [(model["topic"], sum(len(post.get("mentioned", [])) for src in ["reddit","stackoverflow","hf"] for post in model.get(src, []))) for model in mentioned]
# model_mentions.sort(key=lambda x: x[1], reverse=True)

# topics, counts = zip(*model_mentions[:10])

# plt.figure()
# plt.barh(topics, counts)
# plt.title("Top 10 Models by Number of Mentions (Filtered)")
# plt.xlabel("Number of Mentioned Sentences")
# plt.gca().invert_yaxis()
# plt.show()

# # Source distribution before vs. after filtering

# sources = ["reddit", "stack", "hf"]

# united_source_counts = {src: sum(len(model.get(src, [])) for model in united) for src in sources}
# mentioned_source_counts = {src: sum(len(model.get(src, [])) for model in mentioned) for src in sources}

# plt.figure()
# x = range(len(sources))
# plt.bar([i - 0.2 for i in x], united_source_counts.values(), width=0.4, label="United")
# plt.bar([i + 0.2 for i in x], mentioned_source_counts.values(), width=0.4, label="Mentioned")
# plt.xticks(x, sources)
# plt.title("Post Counts per Source (Before vs After Filtering)")
# plt.ylabel("Count")
# plt.legend()
# plt.show()
