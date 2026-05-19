import requests
import json
import time
import sys
from tqdm import tqdm
import random
import praw

# -----------------------------
# Load topics
# -----------------------------
# with open("5-REVIEW_COLLECTION/N_model_dict_dedup_run.json", "r", encoding="utf-8") as f:
#     model_dict = json.load(f)

#after llm check
with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/yes_name.json", "r", encoding="utf-8") as f:
    model_dict = json.load(f)

with open("1-MODEL_FILTERING/N_model_dict.json", "r", encoding="utf-8") as f:
    n_dict = json.load(f)

RRUN = 26 # 24 ten sonrasi yeni run # 26 dan sonrasi llm sonrasi
# TOPICS = [entry["model_name"] for entry in model_dict.values()]
model_names_to_find = list(model_dict.keys())

TOPICS = [
    info["model_id"].replace("/", " ")
    for info in n_dict.values()
    if info["model_name"] in model_names_to_find
][0:]

# 400DEN ONCESI YOK
#posts_10dan sonrakiler gercekten basliyor

reddit = praw.Reddit(client_id='uV3CtS6JqcqFC8vNyJyfaQ',
                     client_secret='iZ7H9lt_bbT7MrhonfslhOdV_l1BLA', password='arda1915',
                     user_agent='PrawTut', username='StreetDonut6976')

url_topics = []

# -----------------------------
# Search loop
# -----------------------------

keywords = [
    "gen ai", "genai", "generative ai",
    "large language model", "LLM", "foundation model",
    "chatbot", "text-to-image", "text-to-video", "image generation", "code generation",
    "ChatGPT", "GPT-4", "GPT-3.5", "Bard AI", "Claude AI",
    "diffusion model", "transformer model", "multimodal AI"
]

or_part = " OR ".join([f'"{kw}"' if " " in kw else kw for kw in keywords])
url_topics = []

for top_num, top in enumerate(tqdm(TOPICS, desc="Fetching topics"), start=1):
    query = f'"{top}" ({or_part})'
    topic_posts = []
    
    consecutive_fail = 0
    max_fail = 5

    while True:
        try:
            # Search Reddit across all subreddits
            for submission in reddit.subreddit("all").search(query, limit=50, sort="relevance"):
                # Get post metadata
                # creation_time = datetime.datetime.fromtimestamp(submission.created_utc) #import datetime
                post_data = {
                    "title": submission.title,
                    "author": str(submission.author),
                    "url": submission.url,
                    "body": submission.selftext if submission.selftext else "[No text — maybe just a link/image]",
                    "subreddit": str(submission.subreddit),
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    # "created_at": creation_date, 
                    "comments": []
                }

                # Fetch comments (flatten "more" comments)
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                    post_data["comments"].append(comment.body)  # only comment text

                topic_posts.append(post_data)

            # Polite delay between topics
            time.sleep(random.uniform(1.5, 3.5))
            break  # exit retry loop if successful

        except Exception as e:
            consecutive_fail += 1
            print(f"Error on topic {top} (attempt {consecutive_fail}): {e}")
            if consecutive_fail >= max_fail:
                print("Max retries reached. Saving progress and exiting.")
                filename = f"5-REVIEW_COLLECTION/reddit_already_united/reddit_posts_{RRUN}_fail_{top_num}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(url_topics, f, indent=2, ensure_ascii=False)
                sys.exit()
            wait_time = 5 * consecutive_fail
            print(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)

    # Only save topic if there is at least one post
    if topic_posts:
        url_topics.append({
            "topic": top,
            "num_posts": len(topic_posts),
            "posts": topic_posts
        })

    # Save progress every 50 topics
    if top_num % 500 == 0:
        filename = f"5-REVIEW_COLLECTION/reddit_already_united/reddit_posts_{RRUN}_check_{top_num}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(url_topics, f, indent=2, ensure_ascii=False)



filename = f"5-REVIEW_COLLECTION/reddit_already_united/reddit_posts_{RRUN}.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(url_topics, f, indent=2, ensure_ascii=False)

print(len(url_topics), "URLs found")
