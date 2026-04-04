# pip install transformers sentence-transformers torch nltk
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          AutoModelForNextSentencePrediction)
import torch, math, re
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
import json
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# 1) Pseudo-perplexity with BERT (MLM)
mlm_name = "bert-base-uncased" 
# FINETUNED BERT WITH 1K REVIEWS
# mlm_name = "./bert-reviews-mlm"

mlm_tok = AutoTokenizer.from_pretrained(mlm_name)
mlm = AutoModelForMaskedLM.from_pretrained(mlm_name).to(DEVICE).eval()

@torch.no_grad()
def pseudo_perplexity(text, max_tokens=256):
    # Truncate to keep runtime reasonable
    enc = mlm_tok(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc["input_ids"][0].to(DEVICE)
    attn = enc["attention_mask"][0].to(DEVICE)

    # Skip special tokens
    mask_token_id = mlm_tok.mask_token_id
    nlls = []
    for i in range(1, input_ids.size(0)-1):
        if attn[i].item() == 0: 
            continue
        orig_id = input_ids[i].item()
        masked = input_ids.clone()
        masked[i] = mask_token_id
        out = mlm(masked.unsqueeze(0))
        logits_i = out.logits[0, i]
        logprob = torch.log_softmax(logits_i, dim=-1)[orig_id]
        nlls.append(-logprob.item())
    if not nlls:
        return float("inf")
    ppl = math.exp(sum(nlls)/len(nlls))
    return ppl

# 2) NSP coherence across adjacent sentences
nsp_name = "bert-base-uncased"  # has NSP head
#FINETUNED BERT WITH 1K REVIEWS
# nsp_name = "./bert-reviews-nsp"

nsp_tok = AutoTokenizer.from_pretrained(nsp_name)
nsp = AutoModelForNextSentencePrediction.from_pretrained(nsp_name).to(DEVICE).eval()

@torch.no_grad()
def nsp_coherence(text, max_pairs=12):
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    pairs = list(zip(sents, sents[1:]))[:max_pairs]
    if not pairs:
        return None
    probs = []
    for a,b in pairs:
        enc = nsp_tok(a, b, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        logits = nsp(**enc).logits[0]
        prob_is_next = torch.softmax(logits, dim=-1)[0].item()
        probs.append(prob_is_next)
    return sum(probs)/len(probs)

# 3) Topical stability via Sentence-BERT embeddings
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

def topical_stability(text):
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if len(sents) < 2:
        return None
    embs = sbert.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
    adj_sim = util.cos_sim(embs[:-1], embs[1:]).diag().mean().item()
    # early/middle/late chunk similarity (robust to long texts)
    thirds = max(1, len(sents)//3)
    a = sbert.encode(" ".join(sents[:thirds]), convert_to_tensor=True, normalize_embeddings=True)
    b = sbert.encode(" ".join(sents[thirds:2*thirds] or sents[thirds:]), convert_to_tensor=True, normalize_embeddings=True)
    c = sbert.encode(" ".join(sents[2*thirds:] or sents[-thirds:]), convert_to_tensor=True, normalize_embeddings=True)
    chunk_sim = (util.cos_sim(a,b).item() + util.cos_sim(b,c).item() + util.cos_sim(a,c).item())/3
    return 0.5*adj_sim + 0.5*chunk_sim

# 4) Tokenization sanity
def tokenization_quality(text):
    enc = mlm_tok(text, return_tensors="pt", truncation=True, max_length=256)
    ids = enc["input_ids"][0].tolist()
    toks = mlm_tok.convert_ids_to_tokens(ids)
    if not toks: return None
    unk_rate = sum(t=="[UNK]" for t in toks)/len(toks)
    # many '##' pieces can indicate odd morphology
    subword_rate = sum(t.startswith("##") for t in toks)/len(toks)
    return 1.0 - min(1.0, 0.7*unk_rate + 0.3*subword_rate)

# Combine into a single 0–100 score
def meaningfulness_score(text):
    # 1) PPL: map lower-better into 0..1 via a soft clamp
    ppl = pseudo_perplexity(text)
    ppl_comp = 1.0 - (min(ppl, 50.0)/50.0)  # 50 is a conservative upper bound
    # 2) NSP: already 0..1
    nsp = nsp_coherence(text) or 0.5
    # 3) Topic stability: cosine ~ 0..1
    topic = topical_stability(text) or 0.5
    # 4) Tokenization quality: 0..1
    tokq = tokenization_quality(text) or 0.5

    # Weighted blend (tweak as you like)
    score01 = 0.40*ppl_comp + 0.25*nsp + 0.25*topic + 0.10*tokq
    return {
        "score_0_100": round(100*score01, 1),
        "components": {
            "pseudo_perplexity": round(ppl, 2),
            "ppl_component": round(ppl_comp, 3),
            "nsp_coherence": round(nsp, 3) if nsp is not None else None,
            "topic_stability": round(topic, 3) if topic is not None else None,
            "tokenization_quality": round(tokq, 3) if tokq is not None else None,
        }
    }

OUTPUT_FOLDER = "5-REVIEW_COLLECTION/united_meaningful"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Make sure you have the required resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Custom stopwords: keep negations
stop_words = set(stopwords.words("english")) - {"not", "no", "nor", "never"}

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercasing
    text = text.lower()
    
    # Remove inline code snippets (between backticks or <code> tags)
    text = re.sub(r"`[^`]+`", " <CODE> ", text)
    text = re.sub(r"<code>.*?</code>", " <CODE> ", text, flags=re.DOTALL)
    
    # Remove URLs, emails, and file paths
    text = re.sub(r"http\S+|www\S+", " <URL> ", text)
    text = re.sub(r"\S+@\S+", " <EMAIL> ", text)
    text = re.sub(r"(/[A-Za-z0-9_\-\.]+)+", " <PATH> ", text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove non-alphabetic tokens, filter stopwords
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return " ".join(tokens)




# --- quick demo ---
if __name__ == "__main__":
    print("doing stuf again.. . ")
    # ################### BASIT TEST BAGZI ORNEKLERLE
    # good = "Photosynthesis converts light energy into chemical energy in plants. Chlorophyll absorbs photons, driving electron transport and ATP synthesis."
    # bad = "florp zingle battery carpet under purple fixes moonlight when 88! run run spaghettify telegraph horse sideways."
    # print("GOOD:", meaningfulness_score(good))
    # print("BAD :", meaningfulness_score(bad))

    # exm1=r"Confirm, the issue still persists Could not load model facebook/bart-large-cnn with any of the following classes: (<class 'transformers.models.bart.modeling_bart.BartForConditionalGeneration'>, <class 'transformers.models.bart.modeling_tf_bart."
    # exm2=r'Hi,I now run it on Google Colab with high RAM T4 GPU in this way summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn",device=0)'
    # exms = ["Instead, I have to do <CODE>.","I use at the moment gemma2 and llama3.1 but for coding I would recommend deepseek.", "More specifically, to go from llama-2 base you could try to pass the weights into the <CODE> script: ``<CODE>`<CODE>training<PATH>LLaMA-2-7B-32K-mqa.sh` for ideas on what parameters you want to use while crunching through the long dataset that @zhangce shared, or try your own long dataset!", "Oto przykład: ``<CODE>`<CODE>`<CODE>`` Natomiast nie mogłem uruchomić standardowego kodu zaproponowanego dla outputu.", "Just got both im-a-good-gpt2-chatbot (model B) and im-also-a-good-gpt2-chatbot (model A) head to head on chat.lmsys.org, imgur gallery 👀", "LLMDet is an advanced system for open-vocabulary object detection that leverages the power of large language models (LLMs) to enable detection of arbitrary object categories, even those not seen during training.", "A (Llama) model between 8B and 70B could fill that place, as could a price reduction on Claude 3 Haiku or Yi-1.5-34B-Chat.", "I think there is a mismatch between the number of tokens in the tokenizer vocab size and config.json When loading the tokenizer: <CODE> we encounter this warning: ``` Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.", "The code is the following:</p> <pre><code> Future&lt;String&gt; getResponse(String message) async { OpenAI.apiKey = openApiKey; try { final chatCompletion = await OpenAI.instance.chat.create( model: 'gpt-3.5-turbo', messages: [ OpenAIChatCompletionChoiceMessageModel( content: message, role: OpenAIChatMessageRole.user, ), ], ); print(chatCompletion); return chatCompletion.choices.first.message.content; } catch (e) { return &quot;Something went wrong.", "Keep the answer concise and ensure that any&quot; &quot;configuration file samples or examples use JSON format.&quot; &quot;\n\n&quot; &quot;{context}&quot; ) qa_prompt = ChatPromptTemplate.from_messages( [ (&quot;system&quot;, system_prompt), MessagesPlaceholder(&quot;chat_history&quot;), (&quot;human&quot;, &quot;{input}&quot;), ] ) question_answer_chain = create_stuff_documents_chain(llm, qa_prompt) rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain) </code></pre> <p>and I'm talking to the microsoft/Phi-3-mini-4k-instruct-gguf model via LM Studio locally with default settings using the local server.</p> <p>For some reason every response from the LLM is prefixed with 'Assistant:' or 'AI:' which is redundant for me, as the user interface will clearly show which text<PATH> is from the AI versus the human messages (think iPhone SMS chat).</p> <p>How can I remove this?" ]
    # print("------ ---- --- - - -- - - - ")
    # for ex in exms:
    #     print("rev :", meaningfulness_score(ex))
    #     print("-     -  - - - - -")
    # ########

    ########## PEARSON COEFF ALMAK ICIN LLMDEN SCOREU ALINAN 500 ORNEK
    ##GET DF
    # Sample dictionary
    with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/check_meaning_output_V0.json", "r", encoding = "utf-8") as f:
        data = json.load(f)

    # Convert dictionary of dictionaries to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')

    # Select only the columns you want
    df = df[['model_id', 'processed', 'output']]
    print("1")
    # Apply the meaningfulness_score function to the 'processed' column
    df['meaningfulness'] = df['processed'].progress_apply(meaningfulness_score)

    # Assuming your DataFrame is called df
    df.to_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/meaningfulness_llm_bert_scores_500_Original.csv", index=False, encoding="utf-8")


















    # ### EN SONDA TUM REVIEWLARI FILTRELEYIP YENI FOLDER ALMAK ICIN
    # MENTIONED_FOLDER = "5-REVIEW_COLLECTION/united_f4"
    # mentioned = load_mentioned_from_folder(MENTIONED_FOLDER)    
    # keys = ["reddit", "hf", "stack"]
    # # 'mentioned' is the list of model dictionaries
    # for model_dict in tqdm(mentioned):
    #     model_id = model_dict.get("model_id", "")
    #     safe_name = re.sub(r"[^\w\-_\.]", "_", model_id)  # sanitize filename
    #     has_reviews = False  # flag to check if any reviews remain
    #     # Process each key
    #     for key in keys:
    #         if key in model_dict:
    #             filtered_reviews = []
    #             for review in model_dict[key]:
    #                 filtered_mentions = []
    #                 for r in review.get("mentioned", []):
    #                     pre_r = preprocess_text(r)
    #                     m_score = meaningfulness_score(pre_r)
    #                     if m_score["score_0_100"] > 50:
    #                         # Keep the review, add score
    #                         filtered_mentions.append({
    #                             "original" : r,
    #                             "text": pre_r,
    #                             "score": m_score["score_0_100"]
    #                         })
    #                 if filtered_mentions:
    #                     # Keep the review with filtered mentions
    #                     review_copy = review.copy()
    #                     review_copy["mentioned"] = filtered_mentions
    #                     filtered_reviews.append(review_copy)

    #             # Replace old reviews with filtered ones
    #             model_dict[key] = filtered_reviews
    #             if filtered_reviews:
    #                 has_reviews = True

    #     # Only save if at least one review remains
    #     if has_reviews:
    #         output_path = os.path.join(OUTPUT_FOLDER, f"{safe_name}.json")
    #         with open(output_path, "w", encoding="utf-8") as f:
    #             json.dump(model_dict, f, ensure_ascii=False, indent=2)

    # print("✅ Done! All processed files saved in:", OUTPUT_FOLDER)