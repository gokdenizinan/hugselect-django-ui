# pip install transformers sentence-transformers torch nltk
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          AutoModelForNextSentencePrediction)
import torch, math, re
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Pseudo-perplexity with BERT (MLM)
mlm_name = "bert-base-uncased"
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

# --- quick demo ---
if __name__ == "__main__":
    good = "Photosynthesis converts light energy into chemical energy in plants. Chlorophyll absorbs photons, driving electron transport and ATP synthesis."
    bad = "florp zingle battery carpet under purple fixes moonlight when 88! run run spaghettify telegraph horse sideways."
    #print("GOOD:", meaningfulness_score(good))
    #print("BAD :", meaningfulness_score(bad))

    exm1=r"Confirm, the issue still persists Could not load model facebook/bart-large-cnn with any of the following classes: (<class 'transformers.models.bart.modeling_bart.BartForConditionalGeneration'>, <class 'transformers.models.bart.modeling_tf_bart."
    exm2=r'Hi,I now run it on Google Colab with high RAM T4 GPU in this way summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn",device=0)'
    exm3="URL hf openchat correct PATH PATH"
    print("rev :", meaningfulness_score(exm3))