import requests
import json
from datetime import datetime, UTC
from tqdm import tqdm
from urllib.parse import quote_plus
import time
import os
# 
# API_KEY = "rl_n1a9FzA9kGditViNpgV9h8jYd" # A0DA
# API_KEY = "rl_f1BgvxfEirh4SW4apVr3ZCTyz" #OPENFIBER
API_KEY = "rl_oNSJpbrjQxw8ytHYBn3mhnxwC" #libcal

# EGER CONNECTION KAYBEDERSEN CHECKPOINTI DOSYASINDAN KENDIN DEGISTIR
# with open("5-REVIEW_COLLECTION/N_model_dict_dedup_run.json", "r", encoding = "utf-8") as f:
#     model_dict = json.load(f)

# with open("5-REVIEW_COLLECTION/stack_disindakiler.json", "r", encoding = "utf-8") as f: # stackte olmayan modellerin reviewlari yeniden runlama
#     disin = json.load(f)

with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/yes_name.json", "r", encoding="utf-8") as f:
    model_dict = json.load(f)

with open("1-MODEL_FILTERING/N_model_dict.json", "r", encoding="utf-8") as f:
    n_dict = json.load(f)

save_nums = list(range(10, 0, -1))
for e in save_nums:
    SAVE_NUM = e
    check_path = f"5-REVIEW_COLLECTION/stackoverflow_already_united/checkpoints_{SAVE_NUM}.json"
    if os.path.exists(check_path):
        with open(check_path, "r", encoding = "utf-8") as f:
            checkpoints = json.load(f)
        break
# run 16 title body olan -> 0dan 4000 e olan topicler
RRUN = SAVE_NUM + 1 # 60 61 62 hepsi kontrol amacli
guven = checkpoints["stack"]
print(f'This is run: {RRUN}/100, from the checkpoint: {checkpoints["stack"]}')

all_results = []
# -----------------------------
# Configuration
# -----------------------------

# TOPICS = [entry["model_name"] for entry in model_dict.values()][checkpoints["stack"]:] #29 k suanki ru
model_names_to_find = list(model_dict.keys())

# TOPICS = [
#     info["model_id"]
#     for info in n_dict.values()
#     if info["model_name"] in model_names_to_find
# ][checkpoints["stack"]:]

TOPICS = [
    info["model_id"].replace("/", " ")
    for info in n_dict.values()
    if info["model_name"] in model_names_to_find
][checkpoints["stack"]:]

# TOPICS = disin[checkpoints["stack"]:] # yarra yedik
# 61den sonrakiler stack_dicsindekiler runiydi
# 73ten sonraliler ise N_dedup_run
PAGE_SIZE = 30 # how many questions per page
MAX_QUESTION_PAGES = 5  # how many pages of questions per topic (adjust)
SLEEP_BETWEEN_REQUESTS = 0.5  # seconds to avoid rate limits
consecutive_errors = 0

# -----------------------------
# Functions
# -----------------------------

def save_and_exit():
    filename = f"5-REVIEW_COLLECTION/stackoverflow_already_united/stackoverflow_{RRUN}_U.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    raise SystemExit(f"❌ Too many failed requests in a row. Results saved to {filename}")

def save_checkpoint(i):
    check_p = {"stack": guven + i-5}
    filename = f"5-REVIEW_COLLECTION/stackoverflow_already_united/checkpoints_{RRUN}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(check_p, f, indent=2, ensure_ascii=False)

def fetch_questions(topic, i):
    global consecutive_errors, all_results
    all_questions = []
    for page in range(1, MAX_QUESTION_PAGES + 1):
        params = {
            "order": "desc",
            "sort": "relevance",
            "q": quote_plus(topic),
            "site": "stackoverflow",
            "pagesize": PAGE_SIZE,
            "page": page,
            "key": API_KEY,
            "filter": "withbody" 
        }

        response = requests.get("https://api.stackexchange.com/2.3/search/advanced", params=params)
        if response.status_code != 200:
            consecutive_errors += 1
            print(f"⚠️ Skipping page {page} for topic {topic} (status {response.status_code}) [{consecutive_errors}/{5}]")
            if consecutive_errors >= 5: #5 TIMES MAX
                if i <= 5:
                    raise SystemExit(f"❌ Too many INITIAL failed requests. Results NOT saved")
                save_checkpoint(i)
                save_and_exit()
            break
        else:
            consecutive_errors = 0  # reset on success

        data = response.json()
        items = data.get("items", [])
        if not items:
            break

        for item in items:
            all_questions.append({
                "searched_topic": topic,
                "question_id": item["question_id"],
                "title": item["title"],
                "body": item.get("body", ""),
                "link": item["link"],
                "creation_date": datetime.fromtimestamp(item["creation_date"], tz=UTC).strftime("%Y-%m-%d %H:%M:%S"),
                "score": item["score"],
                "tags": item["tags"],
                "owner": item.get("owner", {}).get("display_name", "Unknown")
            })
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return all_questions

def fetch_answers(question_id):
    answers_list = []
    params = {
        "order": "desc",
        "sort": "votes",
        "site": "stackoverflow",
        "filter": "withbody",
        "key": API_KEY
    }
    response = requests.get(f"https://api.stackexchange.com/2.3/questions/{question_id}/answers", params=params)
    if response.status_code != 200:
        print(f"⚠️ Skipping answers for question {question_id} (status {response.status_code})")
        return answers_list

    data = response.json()
    for ans in data.get("items", []):
        answers_list.append({
            "answer_id": ans["answer_id"],
            "score": ans["score"],
            "is_accepted": ans.get("is_accepted", False),
            "owner": ans.get("owner", {}).get("display_name", "Unknown"),
            "body": ans["body"]
        })
    return answers_list


# -----------------------------
# Main
# -----------------------------
all_results = []
for i, topic in enumerate(tqdm(TOPICS, desc="Fetching topics"), start=1):
    # if topic in done_topics:
    #     print("skipped")
    #     continue
    # done_topics.append(topic)
    term = topic + " " + "gen ai"
    questions = fetch_questions(topic, i)
    for q in tqdm(questions, desc=f"Fetching answers for {topic}", leave=False):
        q["answers"] = fetch_answers(q["question_id"])
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    all_results.extend(questions)
    # Save every 10 topics
    if i % 100 == 0:
        save_checkpoint(i+5)
        filename = f"5-REVIEW_COLLECTION/stackoverflow_already_united/stackoverflow_{RRUN}_checkpoint_{i}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved checkpoint after {i} topics → {filename}")

# -----------------------------
# Save to JSON
# -----------------------------
with open(f"5-REVIEW_COLLECTION/stackoverflow_already_united/stackoverflow_{RRUN}.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# with open(f"5-REVIEW_COLLECTION/stackoverflow_already_united/done_topics{RRUN}.json", "w", encoding="utf-8") as f:
#     json.dump(done_topics, f, indent=2, ensure_ascii=False)

print(f"✅ Exported Questions: {len(all_results)}")
