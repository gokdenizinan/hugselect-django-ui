import os
import json

# COP # filtre 3te stack model sayisi cok dusuk geldi kontrol etmek istedim
with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/no_name.json", "r", encoding="utf-8") as f:
    model_dict = json.load(f)


with open("1-MODEL_FILTERING/N_model_dict.json", "r", encoding="utf-8") as f:
    n_dict = json.load(f)

model_names_to_find = list(model_dict.keys())

model_ids = [
    info["model_id"]
    for info in n_dict.values()
    if info["model_name"] in model_names_to_find
]

print(len(model_dict))

print(len(n_dict))

print(len(model_names_to_find))

print(len(model_ids))

# 2986
# 71274
# 2986
# 3056 YES MODELID

# 3111
# 71274
# 3111
# 4082 NO MODEL ID

# BUNLAR MENTIONLANAN MODEL IDLER HF VE STACTE 6950

def load_mentioned_from_folder(folder_path):
        mentioned = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        m_reddit = data.get("reddit", [])
                        m_stack = data.get("stack", [])
                        if m_stack or m_reddit:
                            if isinstance(data, list):
                                mentioned.extend(data)
                            else:
                                mentioned.append(data)
                    except json.JSONDecodeError:
                        print(f"⚠️ Could not parse {filename}")
        return mentioned


# MENTIONED_FOLDER = "5-REVIEW_COLLECTION/united_mentioned_reviews"
# mentioned = load_mentioned_from_folder(MENTIONED_FOLDER)