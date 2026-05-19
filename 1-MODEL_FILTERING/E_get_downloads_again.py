# BU FILTRELEME AYARLARINI DEGISTIMEDEN ONCEKI VE SONRAKI MODELLEIN KARSILASTIRILMASI ICIN
#KARSILASTIR VER EKSI VEYA FAZALA MODELLERI hf_model_all_stats VE hf_gone_models DEN CIKART VEYA EKLE SON HALINE GETIR

# import D_check_new_P3
import json

with open("1-MODEL_FILTERING/N_model_dict.json", "r", encoding="utf-8") as f:
    model_dict = json.load(f)

# YENI GONE MODELS ALMAYI UNUTMA DAHA SONRADA KARSILARTIR: 3 TANE HF_GONE MODELS DOSYASI VAR, ARALARINDAN HANGILERI MODEL DICTIN ICINDE ONA BAK
# KODU RUNLAYIP SONUCLARI ALDIKTAN SONRA  hf_model_all_stats ILE BIRLESTIR!!!!
with open("1-MODEL_FILTERING/hf_model_all_stats.json", "r", encoding="utf-8") as f:
    done_dic_1 = json.load(f)

with open("1-MODEL_FILTERING/hf_model_all_stats_different.json", "r", encoding="utf-8") as f:
    done_dic_2 = json.load(f)

with open("1-MODEL_FILTERING/hf_model_all_stats_different_2.json", "r", encoding="utf-8") as f:
    done_dic_3 = json.load(f)

done_ids1 = [d["model_id"] for d in done_dic_1 if "model_id" in d]
done_ids2 = [d["model_id"] for d in done_dic_2 if "model_id" in d]
done_ids3 = [d["model_id"] for d in done_dic_3 if "model_id" in d]
done_ids = done_ids1 + done_ids2 +done_ids3



import json
import time
import random
import requests
from tqdm import tqdm
from huggingface_hub import HfApi

api = HfApi(token="")  

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


new_ids = [info["model_id"] for info in model_dict.values()]
model_ids = [m for m in new_ids if m not in done_ids]

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
            expand=["downloads", "downloadsAllTime", "safetensors", "trendingScore", "likes", "private" , "gated", "library_name", "gguf" , 
                    "config" , "transformersInfo", "cardData", "spaces", "disabled", "usedStorage", "baseModels", "childrenModelCount", 
                    "widgetData", "inference", "xetEnabled", "lastModified", "inferenceProviderMapping",
                    "mask_token", "model-index", "pipeline_tag", "siblings", "resourceGroup", "tags"]
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

        # Extract safetensors info from siblings
        cardData = {}
        if hasattr(info, "cardData") and info.cardData is not None:
            cardData = info.cardData.to_dict()  #

      

        model_stats.append({
            "model_id": model,
            "library_name": getattr(info, "library_name", None),
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "tags": getattr(info, "tags", None),

            "downloads_last_30_days": getattr(info, "downloads", None),
            "downloads_all_time": getattr(info, "downloads_all_time", None),
            "likes": getattr(info, "likes", None),
            "trendingScore": getattr(info, "trendingScore", None),
            "last_modified": info.lastModified.isoformat() if info.lastModified else None,
            "cardData": cardData,

            "private": getattr(info, "private", None),
            "gated": getattr(info, "gated", None),
            "disabled": getattr(info, "disabled", None),

            "usedStorage": getattr(info, "usedStorage", None),
            "tensors": getattr(info, "safetensors", None),
            "config": getattr(info, "config", None),
            "spaces": getattr(info, "spaces", None),
            "transformersInfo": getattr(info, "transformers_info", None),
            
            
            "baseModels": getattr(info, "baseModels", None),
            "childrenModelCount": getattr(info, "childrenModelCount", None),

            "widgetData": getattr(info, "widgetData", None),
            "inference": getattr(info, "inference", None),
            "xetEnabled": getattr(info, "xetEnabled", None),
            "gguf": getattr(info, "gguf", None), ####
            "inferenceProviderMapping": getattr(info, "inferenceProviderMapping", None),
            "resourceGroup": getattr(info, "resourceGroup", None),
            "model_index": getattr(info, "model-index", None),
            "mask_token": getattr(info, "mask_token", None),
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

# --- Save results ---

with open("1-MODEL_FILTERING/hf_model_all_stats_different_3.json", "w", encoding="utf-8") as f:
    json.dump(model_stats, f, indent=2, ensure_ascii=False)

with open("1-MODEL_FILTERING/hf_gone_models_different_3.json", "w", encoding="utf-8") as f:
    json.dump(gone_models, f, indent=2, ensure_ascii=False)

print(f"✅ Collected stats for {len(model_stats)} models, missing {len(gone_models)}")
