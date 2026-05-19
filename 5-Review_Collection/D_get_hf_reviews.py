from huggingface_hub import HfApi
import json 
from tqdm import tqdm
import time
from huggingface_hub import HfApi
import requests


with open("5-REVIEW_COLLECTION/hf_run_again.json", "r", encoding = "utf-8") as f:
    model_dict = json.load(f)

api = HfApi(token="")  
RRUN = 13
reviews = []
gone_models = []

def get_model_list(model_dict):
    model_ids = []
    for item in model_dict:
        model_ids.append(item["model_id"])
    return model_ids

# model_ids = get_model_list(model_dict)
model_ids = model_dict
# model_ids = model_ids[11000:]
# 0 - 250 yenide runlar misin gone dicti yok bunun
# toplanilan model isimlerinin olmadigi liste olustur ve yeniden review topla aminyum

MAX_REQUESTS = 500
WINDOW = 300  # 5 minutes
request_count = 0
window_start = time.time()

def check_rate_limit():
    global request_count, window_start
    request_count += 1
    if request_count >= MAX_REQUESTS:
        elapsed = time.time() - window_start
        if elapsed < WINDOW:
            sleep_time = WINDOW - elapsed
            print(f"⏳ Hit {MAX_REQUESTS} requests in {elapsed:.1f}s — sleeping {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        # reset counter + window
        request_count = 0
        window_start = time.time()


for it, model in enumerate(tqdm(model_ids, desc="Fetching topics"), start = 1):
    # check_rate_limit() 
    repo_id = model 

    try:
        discussions = api.get_repo_discussions(repo_id=repo_id)

        for d in discussions:
            details = api.get_discussion_details(repo_id=repo_id, discussion_num=d.num)
            comments = []
            first = True
            first_body = []
            for index, ev in enumerate(details.events):
                if first:
                    if hasattr(ev, "content"):
                        first_body.append(ev.content)
                        first = False
                else:
                    if hasattr(ev, "content"):
                        comment_dic = {
                            "answer_num" : index,
                            "body" : ev.content
                        }
                        comments.append(comment_dic)

            
            disc_dic = {
                "model_id" : model,
                "disc_num" : d.num,
                "title" : d.title,
                "creation_date" : details.created_at.isoformat(),
                "body" : first_body[0],
                "comments" : comments
            }

            reviews.append(disc_dic)

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Skipping repository {repo_id} due to 404 error: {e}")
            gone_models.append(model)
            continue
        elif e.response.status_code == 403:
            print(f"Skipping repository {repo_id} due to 403 error: {e}")
            gone_models.append(model)
            continue
        else:
            print(f"HTTP error occurred for repository {repo_id}: {e}")
            print(f"SLEEPING FOR {100}")
            time.sleep(100)
        

    if it % 5000 == 0 or it == len(model_ids):
            filename = f"5-REVIEW_COLLECTION/hf_reviews_{RRUN}_check_{it}.json"
            with open(filename, "w", encoding="utf-8") as f:
                    json.dump(reviews, f, indent=2, ensure_ascii=False)
            
            gone_file = f"5-REVIEW_COLLECTION/hf_gones/hf_gone_{RRUN}.json"
            with open(gone_file, "w", encoding="utf-8") as f:
                    json.dump(gone_models, f, indent=2, ensure_ascii=False)


print(f"Successfull collected reviews for: {len(reviews)} Models." )
print("---")
# 1. Count items where body is not empty
non_empty_bodies = sum(1 for item in reviews if item["body"].strip())
# 2. Count total number of comments across all items
total_comments = sum(len(item["comments"]) for item in reviews)
print("Number of non-empty bodies:", non_empty_bodies)
print("Total number of comments:", total_comments)
print("---")
print(f"Total num collected reviews: {non_empty_bodies + total_comments }" )