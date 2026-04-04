import requests
import time
import json
from tqdm import tqdm
from itertools import islice
import copy
from collections import defaultdict
from google import genai
import random
import nltk
import random
from nltk.corpus import stopwords
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

PHASE = 3
print("doing stuff...")

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


# Optional: custom stopwords (if using classical ML)
stop_words = set(stopwords.words("english")) - {"not", "no", "nor", "never"}
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, for_transformer=True):
    if not isinstance(text, str):
        return ""
    
    # Normalize newlines and spacing
    text = text.replace("\n", " ").strip()
    
    # Remove inline code snippets (replace with placeholder)
    text = re.sub(r"`[^`]+`", "<CODE>", text)
    text = re.sub(r"<code>.*?</code>", "<CODE>", text, flags=re.DOTALL)
    
    # Replace URLs, emails, and file paths
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"\b([A-Za-z]:\\|/)[\w\-/\.]+", "<PATH>", text)
    
    if for_transformer:
        # Just basic cleanup — keep punctuation and casing
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    # For traditional ML (TF-IDF, etc.)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

if PHASE == 0:
    with open("1-MODEL_FILTERING/N_model_dict.json", "r", encoding = "utf-8") as f:
        n_dict = json.load(f) #bunu original yap ve yeni mentioned icin path ekle
    
    with open("5-REVIEW_COLLECTION/hf_gones/hf_0_gone_models.json", "r", encoding = "utf-8") as f:
        gone = json.load(f) #bunu original yap ve yeni mentioned icin path ekle

    m_names = [name["model_id"] for name in n_dict.values()]

    for n in n_dict:
        if n in m_names:
            print(n)

if PHASE == 2: 
    OUTPUT_FOLDER = "5-REVIEW_COLLECTION/united_f6"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    MENTIONED_FOLDER = "5-REVIEW_COLLECTION/united_f3"
    mentioned = load_mentioned_from_folder(MENTIONED_FOLDER)
    with open("1-MODEL_FILTERING/N_model_dict.json", "r", encoding = "utf-8") as f:
        n_dict = json.load(f) #bunu original yap ve yeni mentioned icin path ekle

    m_names = [name["model_id"] for name in n_dict.values()]
    in_names = []
    nin_names = []
    for model_dict in tqdm(mentioned):
        
        model_id = model_dict.get("model_id", "")
        safe_name = re.sub(r"[^\w\-_\.]", "__", model_id)  # sanitize filename

        if model_id not in m_names:
            nin_names.append(model_id)
        else:
            in_names.append(model_id)
            output_path = os.path.join(OUTPUT_FOLDER, f"{safe_name}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(model_dict, f, ensure_ascii=False, indent=2)
        # Process each key
    
    # for i in nin_names:
    #     print(i)

    print(m_names[:10])
    print(len(nin_names))
    print(len(in_names))


if PHASE == 3:

    OUTPUT_FOLDER = "5-REVIEW_COLLECTION/united_f7"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # NEW: Folder for models NOT in n_dict
    NON_MENTIONED_FOLDER = "5-REVIEW_COLLECTION/united_f7_not_in_n_dict"
    os.makedirs(NON_MENTIONED_FOLDER, exist_ok=True)

    MENTIONED_FOLDER = "5-REVIEW_COLLECTION/united_f3"
    mentioned = load_mentioned_from_folder(MENTIONED_FOLDER)

    with open("1-MODEL_FILTERING/N_model_dict.json", "r", encoding="utf-8") as f:
        n_dict = json.load(f)

    m_names = [name["model_id"] for name in n_dict.values()]
    in_names = []
    nin_names = []

    for model_dict in tqdm(mentioned):
        model_id = model_dict.get("model_id", "")
        safe_name = re.sub(r"[^\w\-_\.]", "__", model_id)

        if model_id not in m_names:
            nin_names.append(model_id)
            # 💾 Save non-mentioned model in the new folder
            output_path = os.path.join(NON_MENTIONED_FOLDER, f"{safe_name}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(model_dict, f, ensure_ascii=False, indent=2)
        else:
            in_names.append(model_id)
            # 💾 Save mentioned model in the original folder
            output_path = os.path.join(OUTPUT_FOLDER, f"{safe_name}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(model_dict, f, ensure_ascii=False, indent=2)

    print(m_names[:10])
    print("Not in n_dict:", len(nin_names))
    print("In n_dict:", len(in_names))

if PHASE == 1: 

    MENTIONED_FOLDER = "5-REVIEW_COLLECTION/united_mentioned_reviews"
    mentioned = load_mentioned_from_folder(MENTIONED_FOLDER)

    keys = ["reddit", "hf", "stack"]
    for model_dict in tqdm(mentioned):
        model_id = model_dict.get("model_id", "")
        safe_name = re.sub(r"[^\w\-_\.]", "_", model_id)  # sanitize filename
        has_reviews = False  # flag to check if any reviews remain
        # Process each key

        for key in keys:
            if key in model_dict:
                filtered_reviews = []
                for review in model_dict[key]:
                    filtered_mentions = []
                    for r in review.get("mentioned", []):
                        pre_r = preprocess_text(r)

    

