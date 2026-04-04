import requests
import time
import json
from tqdm import tqdm
from itertools import islice
import copy
from collections import defaultdict, Counter
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
import random
import pandas as pd


PHASE = 1
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

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Custom stopwords (keep negations)
stop_words = set(stopwords.words("english")) - {"not", "no", "nor", "never"}
lemmatizer = WordNetLemmatizer()

def generate_model_mentions(model_id: str):
    """
    Given a model ID like 'ZeusLabs/L3-Aethora-15B-V2',
    return a set of possible variants for how it might appear in text.
    """
    variants = set()
    
    # Split the model_id into org and model parts
    if "/" in model_id:
        org, model = model_id.split("/", 1)
    else:
        org, model = "", model_id
    
    # Basic forms
    variants.add(model_id)
    variants.add(model)
    if org:
        variants.add(org)
        variants.add(f"{org} {model}")
        variants.add(f"{org}/{model}")
        variants.add(f"{org}-{model}")
        variants.add(f"{org}:{model}")
        variants.add(f"{org}.{model}")
    
    # Variants with spacing or casing
    variants.add(model.replace("-", " "))
    if org:
        variants.add(f"{org} {model.replace('-', ' ')}")
        variants.add(f"{org}/{model.replace('-', ' ')}")
    
    # Remove duplicates and return sorted for consistency
    sor_vatiants = sorted(variants, key=len, reverse=True)
    return sor_vatiants


def preprocess_text(text, model_id=None):
    """
    Preprocess a text review for AI model training.
    Ensures model_id is preserved even if embedded in paths, code, or URLs.
    """
    if not isinstance(text, str):
        return ""
    ment_names = generate_model_mentions(model_id)
    text = text.strip().replace("\n", " ")

    # --- Step 1: Protect model_id occurrences early ---
    replaced = False
    if model_id:
        safe_model_id = re.escape(model_id)
        # Replace all occurrences (case-insensitive) with a unique placeholder
        text, nu = re.subn(safe_model_id, "<<MODEL_ID>>", text, flags=re.IGNORECASE)
        if nu>0:
            replaced = model_id
                
    for es_id in ment_names:
        # Escape special regex chars and mark it with a placeholder
        safe_model_id = re.escape(es_id)
        text, nu = re.subn(safe_model_id, "<MODEL_ID>", text, flags=re.IGNORECASE)
        if nu > 0:
            replaced = es_id
            break


        # --- Step 2: Replace inline code, URLs, emails, and file paths ---
    # (Note: model_id placeholders won't be affected by these)
    text = re.sub(r"`[^`]+`", "<CODE>", text)
    text = re.sub(r"<code>.*?</code>", "<CODE>", text, flags=re.DOTALL)
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"\b([A-Za-z]:\\|/)[\w\-/\.~]+", "<PATH>", text)
    

    # text = re.sub(r'[\u4e00-\u9fff]+', '<CHINESE>', text)
    # # Hiragana
    # text = re.sub(r'[\u3040-\u309f]+', '<HIRAGANA>', text)
    # # Katakana
    # text = re.sub(r'[\u30a0-\u30ff]+', '<KATAKANA>', text)
    # # Hangul (Korean)
    # text = re.sub(r'[\uac00-\ud7af]+', '<KOREAN>', text)

    # --- Step 3: Restore protected model_id placeholders ---
    if replaced:
        text = text.replace("<<MODEL_ID>>", replaced)
        text = text.replace("<MODEL_ID>", replaced)
    
    # --- Step 4: Light cleanup for transformer or full preprocessing for ML ---

    text = re.sub(r"\s+", " ", text).strip()
    return text

placeholders = {"<URL>", "<CODE>", "<PATH>", "<EMAIL>"} # , '<KOREAN>','<KATAKANA>',"<HIRAGANA>", "<CHINESE>"

def mostly_placeholders(text, threshold=0.9):
    words = text.split()
    if not words:
        return False
    count_placeholders = sum(1 for w in words if w in placeholders)
    return (count_placeholders / len(words)) >= threshold

def filter_reviews(reviews):
    """
    Removes reviews where 'processed' is exactly a placeholder
    or mostly placeholders (>=90% of words).
    """
    filtered = []
    for review in reviews:
        proc = review.get("processed", "")
        if proc in placeholders:
            continue
        if mostly_placeholders(proc):
            continue
        filtered.append(review)
    return filtered


from nltk.tokenize import sent_tokenize

def extract_name_snippet(text, name, max_chars=512):
    """
    Returns a snippet of text (~max_chars) containing the specified name.
    If the sentence is longer than max_chars, centers snippet around the name.
    """
    sentences = sent_tokenize(text)
    
    for idx, sentence in enumerate(sentences):
        if name.lower() in sentence.lower():
            if len(sentence) > max_chars:
                # Find where the name appears
                pos = sentence.lower().find(name.lower())
                
                # Center the window around the name
                start = max(0, pos - max_chars // 2)
                end = min(len(sentence), start + max_chars)
                
                snippet = sentence[start:end].strip()
            else:
                # Normal case: add sentence and following ones if space allows
                snippet = sentence + " "
                next_idx = idx + 1
                while len(snippet) < max_chars and next_idx < len(sentences):
                    snippet += sentences[next_idx] + " "
                    next_idx += 1
                snippet = snippet.strip()
            
            return snippet  # stop after first relevant snippet
    
    return None

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

# Example tokenizer (BERT or RoBERTa)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")  #

def extract_name_snippet_tokens(text, name, max_tokens=512):
    """
    Returns a snippet of text (~max_tokens) containing the specified name.
    Centers snippet around the name if necessary.
    """
    sentences = sent_tokenize(text)
    
    for idx, sentence in enumerate(sentences):
        if name.lower() in sentence.lower():
            # Tokenize sentence
            tokenized = tokenizer.tokenize(sentence)
            num_tokens = len(tokenized)
            
            if num_tokens > max_tokens:
                # Find token index of name
                name_tokens = tokenizer.tokenize(name)
                
                # Find start of first name token in sentence
                for i in range(len(tokenized) - len(name_tokens) + 1):
                    if tokenized[i:i+len(name_tokens)] == name_tokens:
                        name_pos = i
                        break
                else:
                    name_pos = 0  # fallback
                
                # Center window around name tokens
                start = max(0, name_pos - max_tokens // 2)
                end = min(len(tokenized), start + max_tokens)
                
                snippet_tokens = tokenized[start:end]
                snippet = tokenizer.convert_tokens_to_string(snippet_tokens)
            else:
                # Normal case: include following sentences if token space allows
                snippet_tokens = tokenized
                next_idx = idx + 1
                while len(snippet_tokens) < max_tokens and next_idx < len(sentences):
                    next_tokens = tokenizer.tokenize(sentences[next_idx])
                    snippet_tokens.extend(next_tokens)
                    next_idx += 1
                snippet = tokenizer.convert_tokens_to_string(snippet_tokens[:max_tokens])
            
            return snippet  # stop after first relevant snippet
    
    return None

def extract_name_snippet_tokens(text, name, max_tokens=512):
    """
    Returns a snippet of text (~max_tokens) containing the specified name.
    - If name is found: centers snippet around it.
    - If not found: returns snippet from the start of the text.
    """
    sentences = sent_tokenize(text)
    
    for idx, sentence in enumerate(sentences):
        if name.lower() in sentence.lower():
            # Tokenize sentence
            tokenized = tokenizer.tokenize(sentence)
            num_tokens = len(tokenized)
            
            if num_tokens > max_tokens:
                # Find token index of name
                name_tokens = tokenizer.tokenize(name)
                
                # Find start of first name token in sentence
                for i in range(len(tokenized) - len(name_tokens) + 1):
                    if tokenized[i:i+len(name_tokens)] == name_tokens:
                        name_pos = i
                        break
                else:
                    name_pos = 0  # fallback
                
                # Center window around name tokens
                start = max(0, name_pos - max_tokens // 2)
                end = min(len(tokenized), start + max_tokens)
                
                snippet_tokens = tokenized[start:end]
                snippet = tokenizer.convert_tokens_to_string(snippet_tokens)
            else:
                # Normal case: include following sentences if token space allows
                snippet_tokens = tokenized
                next_idx = idx + 1
                while len(snippet_tokens) < max_tokens and next_idx < len(sentences):
                    next_tokens = tokenizer.tokenize(sentences[next_idx])
                    snippet_tokens.extend(next_tokens)
                    next_idx += 1
                snippet = tokenizer.convert_tokens_to_string(snippet_tokens[:max_tokens])
            
            return snippet  # stop after first relevant snippet
    
    # 🔁 Fallback: if name not found, take snippet from start of text
    all_tokens = tokenizer.tokenize(text)
    snippet_tokens = all_tokens[:max_tokens]
    snippet = tokenizer.convert_tokens_to_string(snippet_tokens)
    return snippet


PHASE = 1
if PHASE == 0: 

    MENTIONED_FOLDER = "5-REVIEW_COLLECTION/united_f5"
    mentioned = load_mentioned_from_folder(MENTIONED_FOLDER)

    keys = ["reddit", "hf", "stack"]

    control = []
    total_reviews = 0
    for model_dict in mentioned:
        
        model_id = model_dict.get("model_id", "")
        safe_name = re.sub(r"[^\w\-_\.]", "__", model_id)  # sanitize filename
        for key in keys:
            if key in model_dict:
                filtered_reviews = []
                for review in model_dict[key]:
                    filtered_mentions = []
                    for r in review.get("mentioned", []):
                        total_reviews += 1
                        pairs = {"model_id":"",
                                "original" : "",
                                 "processed" : " "}
                        pre_r = preprocess_text(r, model_id)
                        pairs["model_id"] = model_id
                        pairs["original"] = r
                        pairs["processed"] = pre_r
                        control.append(pairs)
    
    print("TOTAL REVIEWS:" , total_reviews)

    filtered = filter_reviews(control)
    print("FILTERED REVIEWS:" , len(filtered))

    with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/preprocessing_sentiment.json", "w", encoding = "utf-8") as f:
        json.dump(filtered, f , ensure_ascii=False, indent=2)
    

    # The placeholders we care about
 
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('all-MiniLM-L6-v2')


model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')



PHASE = 11

if PHASE ==11:
    # bu promptlu hali llm le grade labeli cikartmak icin # ustteki dictionary alttaki list
    # with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/check_meaning_output_V0.json", "r", encoding = "utf-8") as f:
    #     prompt_f = json.load(f)
    
    with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/preprocessing_sentiment.json", "r", encoding = "utf-8") as f:
        prompt_f = json.load(f)

    rows = []
    # for _, entry in prompt_f.items():
    for entry in prompt_f:
        model_id = entry.get("model_id")
        processed = entry.get("processed")
        original = entry.get("original")
        
        # Extract the integer grade from the "output" field
        output_text = entry.get("output", "")
        match = re.search(r'"grade"\s*:\s*(\d+)', output_text)
        grade = int(match.group(1)) if match else None

        rows.append({
            "model_id": model_id,
            "original" : original,
            "processed": processed,
            # "grade": grade #*10
        })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Identify duplicates based on the 'mentioned' column
    duplicates_mask = df.duplicated(subset=["processed"], keep="first")

    # DataFrame with unique reviews
    df_unique = df[~duplicates_mask].reset_index(drop=True)

    # DataFrame with duplicate reviews
    df_duplicates = df[duplicates_mask].reset_index(drop=True)

    print("Total:", len(df))
    print("Unique:", len(df_unique))
    print("Duplicate:", len(df_duplicates))

    df_unique['snippet'] = df_unique.apply(
    lambda row: extract_name_snippet_tokens(str(row['processed']), row['model_id']),
    axis=1
)
    ref_embeds = model.encode(df_unique["snippet"], normalize_embeddings=True)
    ref_centroid = np.mean(ref_embeds, axis=0)

    review_embeds = model.encode(df_unique['snippet'].tolist(), normalize_embeddings=True)
    similarities = util.cos_sim(review_embeds, ref_centroid)

    df_unique['similarity_score'] = [float(s) for s in similarities]


    threshold = 0.5
    df_unique['is_meaningful'] = df_unique['similarity_score'] > threshold


    # Filter rows where the two columns are different
    diff_rows = df_unique[df_unique["snippet"] != df_unique["processed"]]

    print("diff rows: ", diff_rows)
    print("#######")
    num_true = df_unique['is_meaningful'].sum()
    print("meaningful rows: " , num_true)


    # print(df)
    df_unique.to_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/preprocessing_sentiment_go_new.csv", index=False)

print("done!")
