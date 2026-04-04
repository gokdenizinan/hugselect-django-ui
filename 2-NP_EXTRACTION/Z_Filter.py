import json
import json
import re
import math
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Set


# -----------------------------
# Normalization helpers
# -----------------------------
_punct_re = re.compile(r"[^\w\s/+-]+", re.UNICODE)
_space_re = re.compile(r"\s+")

def normalize_phrase(s: str) -> str:
    """Lowercase, strip, remove most punctuation, collapse whitespace."""
    s = (s or "").strip().lower()
    s = s.replace("\u200b", "")  # zero-width space (sometimes appears)
    s = _punct_re.sub(" ", s)
    s = _space_re.sub(" ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    s = normalize_phrase(s)
    if not s:
        return []
    return s.split()


def normalize_model_id(mid: str) -> str:
    """Replace '__' with '/' to match your canonical ModelID form."""
    return (mid or "").replace("__", "/")


# -----------------------------
# Core filtering configuration
# -----------------------------
CAPABILITY_RE = re.compile(
    r"\b("
    r"generation|support|handling|tuning|trained|training|fine tuned|finetuned|fine-tuned|"
    r"reasoning|solving|classification|translation|retrieval|summarization|explanation|"
    r"instruction|alignment|rlhf|dpo|sft"
    r")\b",
    re.IGNORECASE,
)

# with open("10-EVALUATION/generic_term_500.json", "r", encoding="utf-8") as f:
#     g_terms = json.load(f)

# GENERIC_TERMS = {item["noun_phrase"] for item in g_terms}


# Things that are almost always metadata / tooling / repo artifacts (FP-heavy)
HARD_DROP_EXACT: Set[str] = {
    "allahasfgasdag",
    "more details", "usage", "citation", "documentation", "repository", "limitations", "note", "paper","code","weighting",
    "more information", "repo", "use", "trigger words", "results", "checkpoint", "replicate", "tasks", "citation info", "intended uses",
    "time", "weights", "disclaimer", "usage steps", "project", "bibtex entry", "look", "framework versions", "download model", "model description",
    "files & versions tab", "our work", "team", "details", "Files & versions tab", "original model card", "main binary",
    "latest version", "more code examples", "more traction", "temporary solution", "special thanks",
    "model card", "following code", "main one", "your input", "primary intended use", "primary use", "original paper",
    "other branches", " following results", "detailed results", "your research", "following hyperparameters", "wide range",
    "our paper", "inference", "authors", "users", "following results" , "instructions", "support", "rquirements", "work",
    "thanks", "series", "files", "table", "content", "end", "training details", "number", "docs", "length", "benchmarks",
    "bias", "download", "following error", "free", "overhead", "github", "license", "clone", "transformers", "requirements",
    "unsloth", "risks", "huggingface", "cite", "loss", "finetune", "prompt template", "versions", "access", "contents", 
    "resources", "addition", "outputs", "rights", "need", "credit", "hugging face team", "seperate repo", "subfolder", "your repo",
    }

# HARD_DROP_EXACT = HARD_DROP_EXACT.union(
#     {normalize_phrase(t) for t in GENERIC_TERMS}
# )
# If a phrase contains any of these tokens, drop it (unless you override)
HARD_DROP_TOKENS: Set[str] = {
    "asasfalsajfaf", "versions tab", "specific flags", "=1", "vicuna templates"
    # "repository", "repo", "documentation", "readme", "other hardware", "versions tab"
    # "http", "https", "www",
    # "diffusers", "transformers", "pip", "conda",
    # "gguf", "safetensors", "ckpt", "onnx",  # NOTE: keep onnx only if you want (see WHITELIST)
}

# Whitelist for single-token items you *do* want to keep
SINGLE_TOKEN_WHITELIST: Set[str] = {
    "chatml", "rlhf", "dpo", "sft", "onnx", "lora"
}

# Phrases you want to keep even if they don't match capability regex
PHRASE_WHITELIST: Set[str] = {
    "chatml prompt format",
    "code generation",
    "image generation",
    "text generation",
    "multilingual support",
    "multi language support",
    "fine tuned model",
    "finetuned model",
    "large dataset training",
    "math problem solving",
    "detailed code explanations",
    "code feedback handling",
}


def has_hard_drop_token(tokens: List[str]) -> bool:
    return any(t in HARD_DROP_TOKENS for t in tokens)


# -----------------------------
# Filtering logic for NP entries
# -----------------------------
def filter_np_entries(
    np_info: Dict[str, Dict[str, Any]],
    *,
    min_df: int = 5,
    max_df_ratio: float = 0.9,   # drop phrases that occur in >20% of models
    min_tokens: int = 2,
    max_tokens: int = 6,
    require_capability: bool = True,
    keep_whitelist: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, int]]:
    """
    Returns:
      - filtered_np_info: same structure as input, but only kept NP entries
      - model_to_features: model_id -> list of kept noun phrases (normalized or original kept form)
      - phrase_df: normalized_phrase -> document frequency (#models)
    """

    # 1) Gather model set & compute DF per normalized phrase
    all_models: Set[str] = set()
    phrase_to_models: Dict[str, Set[str]] = defaultdict(set)
    phrase_to_best_original: Dict[str, str] = {}

    for np_key, entry in np_info.items():
        phrase_raw = entry.get("noun_phrase", "")
        phrase_norm = normalize_phrase(phrase_raw)
        if not phrase_norm:
            continue

        mids = entry.get("model_id", []) or []
        mids_normed = [normalize_model_id(m) for m in mids if m]
        for mid in mids_normed:
            all_models.add(mid)
            phrase_to_models[phrase_norm].add(mid)

        # Keep a representative original phrase (first seen)
        phrase_to_best_original.setdefault(phrase_norm, phrase_raw)

    num_models = max(1, len(all_models))
    phrase_df: Dict[str, int] = {p: len(ms) for p, ms in phrase_to_models.items()}
    max_df = int(math.floor(max_df_ratio * num_models))

    # 2) Decide which phrases to keep globally
    keep_phrase: Dict[str, bool] = {}
    for phrase_norm, df in phrase_df.items():
        tokens = tokenize(phrase_norm)
        tok_len = len(tokens)

        # Whitelist can short-circuit
        if keep_whitelist and (phrase_norm in PHRASE_WHITELIST):
            keep_phrase[phrase_norm] = True
            continue

        # Frequency filters
        if df < min_df:
            keep_phrase[phrase_norm] = False
            continue
        if max_df_ratio is not None and max_df > 0 and df > max_df:
            keep_phrase[phrase_norm] = False
            continue

        # Exact hard drop
        if phrase_norm in HARD_DROP_EXACT:
            keep_phrase[phrase_norm] = False
            continue

        # Token-based hard drop
        if has_hard_drop_token(tokens):
            # allow "onnx" etc via whitelist
            if tok_len == 1 and tokens[0] in SINGLE_TOKEN_WHITELIST:
                pass
            else:
                keep_phrase[phrase_norm] = False
                continue

        # Length filters
        if tok_len < min_tokens or tok_len > max_tokens:
            # allow single token whitelist
            if tok_len == 1 and tokens[0] in SINGLE_TOKEN_WHITELIST:
                pass
            else:
                keep_phrase[phrase_norm] = False
                continue

        # # Single-token drop unless whitelisted
        # if tok_len == 1 and tokens[0] not in SINGLE_TOKEN_WHITELIST:
        #     keep_phrase[phrase_norm] = False
        #     continue

        # Capability gating
        if require_capability:
            if CAPABILITY_RE.search(phrase_norm) is None:
                keep_phrase[phrase_norm] = False
                continue

        keep_phrase[phrase_norm] = True

    # 3) Build filtered NP dict (keeping original keys) + model_to_features
    filtered_np_info: Dict[str, Dict[str, Any]] = {}
    model_to_features: Dict[str, List[str]] = defaultdict(list)

    for np_key, entry in np_info.items():
        phrase_raw = entry.get("noun_phrase", "")
        phrase_norm = normalize_phrase(phrase_raw)
        if not phrase_norm:
            continue
        if not keep_phrase.get(phrase_norm, False):
            continue

        # Keep the entry; also normalize model ids inside it
        kept_entry = dict(entry)
        kept_entry["noun_phrase"] = phrase_to_best_original.get(phrase_norm, phrase_raw)

        mids = kept_entry.get("model_id", []) or []
        mids_normed = [normalize_model_id(m) for m in mids if m]
        kept_entry["model_id"] = mids_normed

        filtered_np_info[np_key] = kept_entry

        for mid in mids_normed:
            model_to_features[mid].append(kept_entry["noun_phrase"])

    return filtered_np_info, model_to_features, phrase_df


# -----------------------------
# Optional: cap features per model (Top-K)
# -----------------------------
def score_feature(phrase: str, df: int) -> float:
    """
    Higher is better. Favors:
      - capability phrases
      - rarer phrases
      - reasonable length
    """
    p = normalize_phrase(phrase)
    tokens = tokenize(p)
    length_bonus = 0.0
    if 2 <= len(tokens) <= 5:
        length_bonus = 0.2
    cap_bonus = 0.5 if CAPABILITY_RE.search(p) else 0.0
    rarity = 1.0 / max(1, df)  # rarer => bigger
    return cap_bonus + length_bonus + rarity


from typing import Dict, Any, List, Tuple
from collections import defaultdict

def cap_model_features_np(
    filtered_np_info: Dict[str, Dict[str, Any]],
    model_to_features: Dict[str, List[str]],
    phrase_df: Dict[str, int],
    *,
    topk_by_model: Dict[str, int],
    default_top_k: int = 10,
    dedupe: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """
    Apply dynamic top-K capping per model, and return:
      - capped_np_info (NP-style dict like filter_np_entries)
      - capped_model_to_features (model_id -> capped feature list)
    """

    # 1) Cap features per model using its own K
    capped_model_to_features: Dict[str, List[str]] = {}

    for mid, feats in model_to_features.items():
        k = int(topk_by_model.get(mid, default_top_k))
        if k <= 0:
            capped_model_to_features[mid] = []
            continue

        if not feats:
            capped_model_to_features[mid] = []
            continue

        if dedupe:
            seen = set()
            uniq = []
            for f in feats:
                fn = normalize_phrase(f)
                if fn and fn not in seen:
                    seen.add(fn)
                    uniq.append(f)
            feats = uniq

        scored = []
        for f in feats:
            fn = normalize_phrase(f)
            df = phrase_df.get(fn, 999999)
            scored.append((score_feature(f, df), f))

        scored.sort(reverse=True, key=lambda x: x[0])
        capped_model_to_features[mid] = [f for _, f in scored[:k]]

    # 2) Build allowed (model, phrase_norm) pairs
    allowed = set()
    for mid, feats in capped_model_to_features.items():
        for f in feats:
            allowed.add((mid, normalize_phrase(f)))

    # 3) Rebuild NP dict keeping only surviving model associations
    capped_np_info: Dict[str, Dict[str, Any]] = {}
    for np_key, entry in filtered_np_info.items():
        phrase = entry.get("noun_phrase", "")
        phrase_norm = normalize_phrase(phrase)

        mids = entry.get("model_id", []) or []
        kept_mids = [mid for mid in mids if (mid, phrase_norm) in allowed]

        if not kept_mids:
            continue

        new_entry = dict(entry)
        new_entry["model_id"] = kept_mids
        capped_np_info[np_key] = new_entry

    return capped_np_info, capped_model_to_features



import math
from typing import Dict

def choose_topk_log_scaled(
    model_desc_len: Dict[str, int],
    *,
    k_min: int = 5,
    k_max: int = 25,
    k_mid: int = 10,
) -> Dict[str, int]:
    """
    Choose top_k per model using log-scaled length.
    Designed for strong monotonic but imperfect linear correlation.
    """

    # Collect valid lengths
    lengths = sorted(
        [v for v in model_desc_len.values() if isinstance(v, (int, float)) and v > 0]
    )
    if not lengths:
        return {m: k_mid for m in model_desc_len}

    def pct(p: float) -> float:
        i = int(round(p * (len(lengths) - 1)))
        return lengths[max(0, min(len(lengths) - 1, i))]

    L10 = pct(0.10)
    L50 = pct(0.50)
    L90 = pct(0.90)

    log_L10 = math.log(L10)
    log_L50 = math.log(L50)
    log_L90 = math.log(L90)

    out: Dict[str, int] = {}

    for m, L in model_desc_len.items():
        if not isinstance(L, (int, float)) or L <= 0:
            out[m] = k_mid
            continue

        x = math.log(L)

        if x <= log_L50:
            # interpolate between (L10 → ~6) and (L50 → 10)
            t = (x - log_L10) / max(1e-6, (log_L50 - log_L10))
            k = 6 + t * (k_mid - 6)
        else:
            # interpolate between (L50 → 10) and (L90 → ~20)
            t = (x - log_L50) / max(1e-6, (log_L90 - log_L50))
            k = k_mid + t * (20 - k_mid)

        k = int(round(k))
        k = max(k_min, min(k_max, k))
        out[m] = k

    return out


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Load your NP dictionary JSON (the structure you showed)
    # np_info.json should be like: {"NP_2": {...}, "NP_3": {...}, ...}
    # INPUT_JSON = "2-NP_EXTRACTION/NP_global_dictionary_comfy.json"
    INPUT_JSON = "2-NP_EXTRACTION/NP_comfy_P.json"
    OUT_NP_JSON = "2-NP_EXTRACTION/NP_comfy_PP.json"
    OUT_NP_LIST_JSON = "2-NP_EXTRACTION/NP_comfy_PP_list.json"
    OUT_MODEL_FEATURES_JSON = "2-NP_EXTRACTION/NP_compfy_B.json"
    OUT_MODEL_FEATURES_CAPPED_JSON = "2-NP_EXTRACTION/NP_compfy_C.json"


    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        np_info = json.load(f)

    filtered_np_info, model_to_features, phrase_df = filter_np_entries(
        np_info,
        min_df=0,
        max_df_ratio=0.5,      # add the missing MAX frequency filter
        min_tokens=0,
        max_tokens=5,
        require_capability=False # capability gating
    )
    phrase_list = [
    normalize_phrase(entry["noun_phrase"])
    for entry in filtered_np_info.values()
    if entry.get("noun_phrase")
]

    with open(OUT_NP_LIST_JSON, "w", encoding="utf-8") as f:
        json.dump(phrase_list, f, ensure_ascii=False, indent=2)

    with open("10-EVALUATION/modelid_to_description_length.json", "r", encoding="utf-8") as f:
        model_desc_len = json.load(f)

    # model_desc_len: Dict[str,int]  (model_id -> description length)
    topk_by_model = choose_topk_log_scaled(model_desc_len)

    capped_np_info, model_to_features_capped = cap_model_features_np(
        filtered_np_info,
        model_to_features,
        phrase_df,
        topk_by_model=topk_by_model,
        default_top_k=10,
        dedupe=True,
    )

    # print("Example Ks:", list(topk_by_model.items())[:10])
    # print("Min/Max K:", min(topk_by_model.values()), max(topk_by_model.values()))


    with open(OUT_NP_JSON, "w", encoding="utf-8") as f:
        json.dump(filtered_np_info, f, ensure_ascii=False, indent=2)

    with open(OUT_MODEL_FEATURES_JSON, "w", encoding="utf-8") as f:
        json.dump(model_to_features, f, ensure_ascii=False, indent=2)

    with open(OUT_MODEL_FEATURES_CAPPED_JSON, "w", encoding="utf-8") as f:
        json.dump(capped_np_info, f, ensure_ascii=False, indent=2)

    print("Input NP entries:", len(np_info))
    print("Filtered NP entries:", len(filtered_np_info))
    print("Models with capped features:", len(capped_np_info))
    print("Saved:", OUT_NP_JSON, OUT_MODEL_FEATURES_JSON, OUT_MODEL_FEATURES_CAPPED_JSON)


