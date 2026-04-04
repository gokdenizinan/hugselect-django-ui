import os
import json
import time
import random
import re
from collections import defaultdict
from placeholder_terms import PLACEHOLDER_PATTERNS
from datetime import datetime
from tqdm import tqdm


## BU KODUN AMACI DOWNLOAD SAYISI FILTRESINI ENTEGRE ETMEK AYNI ZAMANDA DUPLICATE FILTRESINI GRAFIKLE GOSTERMEK
# Start the timer
start_time = time.time()

# # Set your directory path
# # Set a random seed for reproducibility
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)

# # List to store results
# selected_files = []
# author_to_files = defaultdict(list)

# # Set your directory path
# directory = "HF-Models-Y1"


# with open("1-MODEL_FILTERING/N_duplicate_model_names.json", "r", encoding = "utf-8") as f:
#     duplicate = json.load(f)

# with open("1-MODEL_FILTERING/hf_model_download_stats.json", "r", encoding = "utf-8") as f:
#     p4_download = json.load(f)


# def has_description(data):
#     description = data.get("description", None)
#     if not description or len(description.strip()) < 100: # too short
#         return True
#     elif len(set(description.split())) < 30: # too short or generic
#         if description.count("model") > 5:
#             return True
#         elif len(set(description.split())) < 20:
#             return True    
    
#     desc = description.lower().strip()
#     for pattern in PLACEHOLDER_PATTERNS:
#         if re.search(pattern, desc):
#             return True
#     return False

# def has_files(data):
#     files = data.get("file_count", None)
#     return files <=1

# def has_pipetag(data):
#     likes = data.get("likes", 0)
    
#     tag = data.get("pipeline_tag", None)
#     if likes < 20:
#         return tag == [] or tag is None
#     else:
#         return False


# def download_filter(models, m_all = 50, m_30 = 3):
#     model_ids = [
#         name
#         for model in models
#         for name, stats in model.items()
#         if stats["downloads_all_time"] < m_all
#         and stats["downloads_last_30_days"] < m_30
#     ]
#     return model_ids

# lame_models = download_filter(p4_download)

# def has_downloads(data):
#     global lame_models, lamer_models
#     likes =  data.get("likes", 0)
#     last_modified = data.get("lastModified", None)
#     model_id = data.get("modelId", "")
#     if likes > 50:
#         return False  # Do NOT filter out popular models
#     elif likes < 1:
#         if last_modified:
#             try:
#                 last_modified_dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
#                 # Automatically pass if modified after 2024
#                 if last_modified_dt.year > 2024:
#                     if model_id in lamer_models:
#                         return True
#                     return False  # Do NOT filter out
#             except Exception:
#                 pass  # ignore bad format
#         # Filter only models with zero likes for old models
#     else:
#         if model_id in lame_models:
#             return True
#     return False

# def has_duplicate(data):
#     model_id = data.get("modelId", "")
#     return model_id in duplicate


# def has_2025(data):
#     dondur = {}
#     last_modified = data.get("lastModified", None)
#     likes = data.get("likes", 0)
#     model_id = data.get("modelId", "")
#     last_modified_dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
#     if last_modified_dt.year > 2024 and likes == 0:
#         dondur[model_id] = last_modified_dt.year

#     return dondur

# print(f"Checking directory: {directory}")

# # Get all .json files in the directory
# all_files = [f for f in os.listdir(directory) if f.endswith(".json")]

# # Sample 1% of files (at least 1 file)
# SAMPLE = 1 # 0.01
# sample_size = max(1, int(len(all_files) * SAMPLE))
# sampled_files = random.sample(all_files, sample_size)

# print(f"Randomly selected {sample_size} of {len(all_files)} files (seed={RANDOM_SEED}).\n")

# models_2025 = []
# # Loop through sampled files
# for filename in tqdm(sampled_files):
#     file_path = os.path.join(directory, filename)
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         if not has_pipetag(data) and not has_description(data) and not has_files(data):
#             if not has_duplicate(data):
#                 if has_2025(data):
#                     models_2025.append(filename)

# with open("1-MODEL_FILTERING/models_2025.json", "w", encoding="utf-8") as f:
#     json.dump(models_2025, f, indent=2, ensure_ascii=False)

# print(len(models_2025))
# print("oncesinde 430k idi bunun 10-20k ya falan inmesi lazim aga")

################################
################################
################################
################################

with open("1-MODEL_FILTERING/models_2025.json", "r", encoding = "utf-8") as f:
    models_2025 = json.load(f)

import json
import time
import random
import requests
from tqdm import tqdm
from huggingface_hub import HfApi

api = HfApi(token="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")  

# --- Retry wrapper ---
def safe_request(fn, *args, **kwargs):
    retries = 3
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [429, 500, 502, 503]:
                sleep_time = (2 ** attempt) + random.random()
                print(f"Retrying after {sleep_time:.2f}s due to {e}")
                time.sleep(sleep_time)
            else:
                raise
    raise RuntimeError("Max retries reached")

# --- Simple request limiter ---
MAX_REQUESTS = 950
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

# --- Load models ---
model_ids = models_2025

# --- Fetch info ---
model_stats = []
gone_models = []
anan =True

for model in tqdm(model_ids, desc="Fetching model info"):
    check_rate_limit() 
    try:
        info = safe_request(
            api.model_info,
            repo_id=model,
            expand=["downloads", "downloadsAllTime"]
        )

        if anan == True:
                    # Using __dict__ to get instance attributes
            for attr, value in info.__dict__.items():
                print(f"{attr}: {value}")

            # Or using dir() to get all attributes including methods (may include built-ins)
            for attr in dir(info):
                if not attr.startswith("__"):
                    print(f"{attr}: {getattr(info, attr)}")
            anan = False

      

        model_stats.append({
            "model_id": model,
            "downloads_last_30_days": getattr(info, "downloads", None),
            "downloads_all_time": getattr(info, "downloads_all_time", None),
        })

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            gone_models.append(model)
        else:
            print(f"HTTP error for {model}: {e}")
        continue
    except Exception as e:
        print(f"Unexpected error for {model}: {e}")
        continue


download_dic = []

for model in model_stats:
    d_dic = {model["model_id"] : 
             {
        "downloads_all_time" : model["downloads_all_time"],
        "downloads_last_30_days" : model["downloads_last_30_days"],  
    }}
    download_dic.append(d_dic)
# --- Save results ---

with open("1-MODEL_FILTERING/hf_model_download_stats_2024.json", "w", encoding="utf-8") as f:
    json.dump(download_dic, f, indent=2, ensure_ascii=False)

with open("1-MODEL_FILTERING/hf_gone_models_2024.json", "w", encoding="utf-8") as f:
    json.dump(gone_models, f, indent=2, ensure_ascii=False)

print(f"✅ Collected stats for {len(model_stats)} models, missing {len(gone_models)}")
