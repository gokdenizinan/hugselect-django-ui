import json

import re
import json
import unicodedata
from collections import OrderedDict
from collections import Counter

# ----------------------------
# Normalization helpers
# ----------------------------

import re

# --- token classifiers ---

# Pure all-caps acronyms: NLP, DPO, RLHF
_ALL_CAPS = re.compile(r"^[A-Z]{2,}$")

# All-caps with trailing + or ++: RLHF+, DPO++, PPO+
_ALL_CAPS_PLUS = re.compile(r"^[A-Z]{2,}\+{1,3}$")

# Common programming-ish tokens: C#, F#, C++ (simple handling)
_LANG_TOKENS = re.compile(r"^(?:[A-Z]\+{2}|[A-Z]#)$")  # C++, C#, F#

# Mixed-case acronym-ish tokens like LoRA, QLoRA, xLoRA, mLoRA, ReAct, ChatML
# Heuristic: contains >=2 uppercase letters AND at least one lowercase letter (CamelCase-ish)
_MIXED_ACRONYMISH = re.compile(r"^(?=.*[A-Z].*[A-Z])(?=.*[a-z])[A-Za-z]+$")

# Model/tech patterns with digits/hyphens/dots often should keep case:
# GPT-4, Llama-2, Mixtral-8x7B, BERT-base, gpt-4o-mini
_TECH_HYPHENATED = re.compile(r"^[A-Za-z]+[A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+$")

# Alphanumeric with capitals (FP16, BF16, INT8, 7B, 8x7B, 4o, A100)
_ALNUM_TECH = re.compile(r"^(?=.*[A-Z])(?=.*\d)[A-Za-z0-9]+$|^\d+(?:x\d+)?[A-Za-z]+$")


def is_protected_token(tok: str) -> bool:
    """
    Tokens that should NOT be lowercased because casing carries meaning / is conventional in ML/tech text.
    Covers:
      - RLHF+, DPO++, PPO+
      - LoRA, QLoRA, ReAct, ChatML
      - GPT-4, Llama-2, BERT-base, gpt-4o-mini (keeps as-is)
      - FP16, BF16, INT8, 7B, 8x7B
      - C++, C#
    """
    if not tok:
        return False

    # Common surrounding punctuation: keep inner token; caller tokenization may already do this,
    # but just in case.
    t = tok.strip()

    return bool(
        _ALL_CAPS.match(t)
        or _ALL_CAPS_PLUS.match(t)
        or _LANG_TOKENS.match(t)
        or _MIXED_ACRONYMISH.match(t)
        or _TECH_HYPHENATED.match(t)
        or _ALNUM_TECH.match(t)
    )


def contains_acronym_or_tech(toks: list[str]) -> bool:
    return any(is_protected_token(t) for t in toks)


_ACRONYM_RE = re.compile(r"^[A-Z]{2,}$")  # basic: NLP, DPO, GPU
_DIGIT_RE = re.compile(r"\d")

def is_acronym_token(tok: str) -> bool:
    """
    Detect acronym-ish tokens (all caps, length>=2).
    Conservative on purpose.
    """
    return bool(_ACRONYM_RE.match(tok))

def normalize_text_basic(s: str) -> str:
    """
    Unicode normalize, normalize hyphens/quotes, collapse whitespace.
    """
    s = unicodedata.normalize("NFKC", s)

    # Normalize common dash variants to hyphen
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")

    # Turn hyphens/underscores into spaces for variant matching
    s = re.sub(r"[-_]+", " ", s)

    # Strip "outer" punctuation while keeping inner word chars/numbers
    # (We handle token-level cleanup below.)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize_phrase(s: str) -> list[str]:
    """
    Tokenize into "word-like" pieces: letters/numbers plus some technical joiners.
    Keeps tokens like: C++, C#, node.js, BERT-base, gpt-4o-mini reasonably intact.
    """
    # Replace most punctuation with spaces, but keep . + # / in tokens
    # and keep apostrophes inside words.
    cleaned = re.sub(r"[^\w\.\+#/\' ]+", " ", s, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []
    return cleaned.split(" ")

def singularize_english_noun(word: str) -> str:
    """
    Very careful, heuristic 'lemmatization' for head nouns.
    Only applies to simple plural forms and avoids common traps.
    """
    w = word
    if not w.isalpha():
        return w
    if len(w) < 4:
        return w

    lw = w.lower()

    # Don't singularize words that often end in 's' but aren't plurals
    # (extend as needed for your domain)
    NO_SINGULARIZE = {
        "news", "series", "species", "physics", "ethics", "mathematics",
    }
    if lw in NO_SINGULARIZE:
        return w

    # ies -> y (datasets? not; but "companies" -> "company")
    if lw.endswith("ies") and len(lw) > 4:
        return w[:-3] + ("y" if w[-3:].islower() else "Y")

    # es -> e/'' for some endings, but keep conservative
    if lw.endswith("es") and len(lw) > 4:
        # e.g., "boxes" -> "box", "watches" -> "watch"
        if lw.endswith(("sses", "shes", "ches", "xes", "zes")):
            return w[:-2]

    # simple trailing s
    if lw.endswith("s") and not lw.endswith("ss"):
        return w[:-1]

    return w

def canonicalize_noun_phrase(np: str) -> tuple[str, dict]:
    s = normalize_text_basic(np)
    toks = tokenize_phrase(s)
    if not toks:
        return "", {"tokens": [], "num_tokens": 0, "has_digits": False, "has_acronym": False}

    has_digits = any(ch.isdigit() for ch in s)
    out = []

    for t in toks:
        if is_protected_token(t):
            out.append(t)          # keep exact casing
        else:
            out.append(t.lower())  # normalize normal words

    # head noun singularization: only if last token is NOT protected and is alpha
    head_raw = toks[-1]
    head = out[-1]
    if (not is_protected_token(head_raw)) and head.isalpha():
        out[-1] = singularize_english_noun(head)

    canon = " ".join(out)
    canon = re.sub(r"\s+", " ", canon).strip()

    feats = {
        "tokens": out,
        "num_tokens": len(out),
        "has_digits": has_digits,
        "has_acronym": contains_acronym_or_tech(toks),
    }
    return canon, feats

def dedupe_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        # treat dict/list as unhashable -> stringify for stable dedupe if needed
        key = x if isinstance(x, (str, int, float, bool, type(None))) else json.dumps(x, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out

# ----------------------------
# Merge logic
# ----------------------------

LIST_FIELDS_DEFAULT = {"model_id", "sentence","merged_from"}

def merge_entries(base: dict, incoming: dict, source_key: str | None, list_fields: set[str]) -> dict:
    if source_key:
        base.setdefault("source_keys", [])
        base["source_keys"].append(source_key)
        base["source_keys"] = dedupe_preserve_order(base["source_keys"])

    # Always merge incoming source_keys
    if "source_keys" in incoming:
        base.setdefault("source_keys", [])
        base["source_keys"].extend(incoming["source_keys"])
        base["source_keys"] = dedupe_preserve_order(base["source_keys"])

    # count
    if "count" in incoming:
        base["count"] = int(base.get("count", 0)) + int(incoming["count"])

    if "entity_labels" in incoming:
        base.setdefault("entity_labels", Counter())
        for k, v in incoming["entity_labels"].items():
            base["entity_labels"][k] += v

    # merge list fields
    for f in list_fields:
        if f in incoming:
            inc_val = incoming[f]
            if inc_val is None:
                continue
            if not isinstance(inc_val, list):
                inc_val = [inc_val]
            base.setdefault(f, [])
            base[f].extend(inc_val)
            base[f] = dedupe_preserve_order(base[f])

    return base


def process_np_dict(np_dict: dict,
                    merged_json_path: str = "noun_phrases_merged.json",
                    phrase_list_json_path: str = "noun_phrases_list.json",
                    list_fields: set[str] | None = None):
    """
    Main pipeline.
    np_dict: your dict-of-dicts keyed by NP_1, NP_2, etc.
    Saves:
      - merged_json_path (full merged structure)
      - phrase_list_json_path (list of canonical noun phrases)
    """
    if list_fields is None:
        # discover list fields dynamically (safe fallback)
        list_fields = set(LIST_FIELDS_DEFAULT)
        for _, entry in np_dict.items():
            for k, v in entry.items():
                if isinstance(v, list):
                    list_fields.add(k)

    merged = {}              # canonical_phrase -> merged_entry
    features = {}            # canonical_phrase -> feats for sorting

    for key, entry in np_dict.items():
        raw_np = entry.get("noun_phrase", "")
        canon_np, feats = canonicalize_noun_phrase(raw_np)
        if not canon_np:
            continue

        if canon_np not in merged:
            merged[canon_np] = {
                "noun_phrase": canon_np,
                "count": int(entry.get("count", 0)) if isinstance(entry.get("count", 0), (int, float)) else 0,
                "source_keys": [key],
            }
            # initialize lists
            for f in list_fields:
                if f in entry:
                    v = entry[f]
                    merged[canon_np][f] = v if isinstance(v, list) else [v]
                    merged[canon_np][f] = dedupe_preserve_order(merged[canon_np][f])

            # keep a couple useful originals if you want traceability
            merged[canon_np]["noun_phrase_raw_examples"] = dedupe_preserve_order([raw_np])

            features[canon_np] = feats
        else:
            merged[canon_np]["noun_phrase_raw_examples"].append(raw_np)
            merged[canon_np]["noun_phrase_raw_examples"] = dedupe_preserve_order(
                merged[canon_np]["noun_phrase_raw_examples"]
            )
            merged[canon_np] = merge_entries(merged[canon_np], entry, key, list_fields)

        # ensure no repeats in all list fields after every merge
        for f in list_fields:
            if f in merged[canon_np]:
                merged[canon_np][f] = dedupe_preserve_order(merged[canon_np][f])

    # Sort:
    # 1) count ascending (rarer first)
    # 2) phrase length descending (longer first)
    # 3) technical: contains digits/acronyms descending
    def sort_key(item):
        canon_np, data = item
        feats = features.get(canon_np, {})
        count = int(data.get("count", 0))
        num_tokens = int(feats.get("num_tokens", len(canon_np.split())))
        # technical score: digits + acronym tokens
        tech_score = int(feats.get("has_digits", False)) + int(feats.get("has_acronym", False))
        return (count, -num_tokens, -tech_score, canon_np)

    merged_items_sorted = sorted(merged.items(), key=sort_key)

    # Ordered dict output
    # Ordered dict output with numeric string keys
    merged_ordered = OrderedDict()
    for i, (canon_np, data) in enumerate(merged_items_sorted):
        # final dedupe safety
        for f in list_fields:
            if f in data and isinstance(data[f], list):
                data[f] = dedupe_preserve_order(data[f])

        merged_ordered[str(i)] = data


    # Save merged
    with open(merged_json_path, "w", encoding="utf-8") as f:
        json.dump(merged_ordered, f, ensure_ascii=False, indent=2)

    # Save phrase list
    phrase_list = [canon_np for canon_np, _ in merged_items_sorted]
    with open(phrase_list_json_path, "w", encoding="utf-8") as f:
        json.dump(phrase_list, f, ensure_ascii=False, indent=2)

    return merged_ordered, phrase_list

from collections import OrderedDict
from rapidfuzz import process, fuzz

def _norm_np(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

import re
from collections import OrderedDict
from rapidfuzz import process, fuzz

_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")

def _np_alnum_len(s: str) -> int:
    """Count letters/digits ignoring spaces/punct."""
    return sum(len(m.group(0)) for m in _ALNUM_RE.finditer(s or ""))

def _np_num_tokens(s: str) -> int:
    return len((s or "").split())

def conservative_threshold(phrase: str,
                           base: int = 92,
                           min_threshold: int = 90,
                           max_threshold: int = 99) -> int:
    """
    Adaptive threshold: shorter phrases require higher similarity to merge.
    Tune base/min/max as needed.
    """
    L = _np_alnum_len(phrase)
    T = _np_num_tokens(phrase)

    # Very short: almost never merge unless it's basically identical
    if T == 1:
        if L <= 4:   # "RAG", "NLP", "tool", "data"
            return 99
        if L <= 6:
            return 98
        if L <= 8:
            return 96

    if T == 2:
        if L <= 8:
            return 98
        if L <= 12:
            return 96

    # Moderate length: slightly stricter than base
    if L <= 16:
        return min(max_threshold, max(base, 94))

    # Longer phrases: base is fine
    return max(min_threshold, base)

def fuzzy_merge_after_canonical_conservative(
    merged_ordered: dict,
    base_threshold: int = 90,          # used for longer phrases
    anchor_min_count: int = 5,
    block_tokens: int = 2,
    list_fields: set[str] | None = None,
    allow_merge_single_token: bool = True,
) -> OrderedDict:
    """
    Post-canonical fuzzy merge with conservative behavior for short noun phrases.
    Uses an adaptive threshold that is higher for short phrases.
    """
    if not merged_ordered:
        return OrderedDict()

    # Discover list fields if not provided
    if list_fields is None:
        list_fields = set(LIST_FIELDS_DEFAULT)
        for _, entry in merged_ordered.items():
            for k, v in entry.items():
                if isinstance(v, list):
                    list_fields.add(k)

    # phrase -> entry
    phrase_to_entry = {}
    for _, entry in merged_ordered.items():
        p = (entry.get("noun_phrase") or "").strip()
        if p:
            phrase_to_entry[p] = entry

    def norm(s: str) -> str:
        return _norm_np(s)  # your existing normalizer

    phrases = list(phrase_to_entry.keys())

    anchors = [p for p in phrases if int(phrase_to_entry[p].get("count", 0)) >= int(anchor_min_count)]
    if not anchors:
        anchors = phrases[:]

    # Precompute norms
    norm_cache = {p: norm(p) for p in phrases}
    anchor_norm = {p: norm_cache[p] for p in anchors}

    # Buckets by first N tokens of normalized anchor phrase
    buckets: dict[str, list[str]] = {}
    for a in anchors:
        key = " ".join(anchor_norm[a].split()[:block_tokens])
        buckets.setdefault(key, []).append(a)

    anchor_set = set(anchors)
    candidates = sorted(
        (p for p in phrases if p not in anchor_set),
        key=lambda p: int(phrase_to_entry[p].get("count", 0))
    )

    removed = set()

    for p in candidates:
        if p in removed:
            continue

        # Optional: refuse merging very short single-token phrases altogether
        if not allow_merge_single_token and _np_num_tokens(p) == 1:
            continue

        p_norm = norm_cache[p]
        key = " ".join(p_norm.split()[:block_tokens])
        cand_anchors = buckets.get(key) or anchors

        # Choose scorer: safer for short strings
        L = _np_alnum_len(p)
        scorer = fuzz.WRatio if L <= 12 else fuzz.ratio

        # Adaptive threshold for this phrase
        thr = conservative_threshold(p, base=base_threshold)

        cand_anchor_norms = [anchor_norm[a] for a in cand_anchors]

        m = process.extractOne(
            p_norm,
            cand_anchor_norms,
            scorer=scorer,
            score_cutoff=thr,
        )
        if m is None:
            continue

        _, score, idx = m
        best_anchor_phrase = cand_anchors[idx]
        if best_anchor_phrase == p:
            continue

        base = phrase_to_entry[best_anchor_phrase]
        incoming = phrase_to_entry[p]

        # Record phrase-level merge provenance
        base.setdefault("merged_from", [])
        base["merged_from"].append(p)
        base["merged_from"] = dedupe_preserve_order(base["merged_from"])

        # Merge using your helper (preserves list fields, sums count, etc.)
        phrase_to_entry[best_anchor_phrase] = merge_entries(
            base=base,
            incoming=incoming,
            source_key=None,
            list_fields=list_fields,
        )

        # Merge raw examples too
        if "noun_phrase_raw_examples" in incoming:
            base.setdefault("noun_phrase_raw_examples", [])
            base["noun_phrase_raw_examples"].extend(incoming.get("noun_phrase_raw_examples", []))
            base["noun_phrase_raw_examples"] = dedupe_preserve_order(base["noun_phrase_raw_examples"])

        removed.add(p)
        del phrase_to_entry[p]

    # Repack into OrderedDict with numeric keys (keep your previous sorting)
    def tech_score_for_phrase(phrase: str) -> int:
        toks = phrase.split()
        has_digits = any(any(ch.isdigit() for ch in t) for t in toks)
        has_acr = contains_acronym_or_tech(toks)
        return int(has_digits) + int(has_acr)

    items = list(phrase_to_entry.items())
    items.sort(
        key=lambda kv: (
            int(kv[1].get("count", 0)),
            -len(kv[0].split()),
            -tech_score_for_phrase(kv[0]),
            kv[0],
        )
    )

    out = OrderedDict()
    for i, (phrase, entry) in enumerate(items):
        for f in list_fields:
            if f in entry and isinstance(entry[f], list):
                entry[f] = dedupe_preserve_order(entry[f])
        entry["noun_phrase"] = phrase
        out[str(i)] = entry

    return out

import re
from collections import OrderedDict

_AUTHOR_SPLIT_RE = re.compile(r"(__|/)", re.UNICODE)

def model_id_to_author(model_id: str) -> str:
    """
    Author is the part of the model_id before '__' or '/'.
    Examples:
      'openai__gpt-4'      -> 'openai'
      'meta/llama-2-7b'    -> 'meta'
      'together/xyz__v2'   -> 'together'
      'mistral'            -> 'mistral'
    """
    if not model_id:
        return ""
    s = str(model_id).strip()
    if not s:
        return ""
    # split on first occurrence of '__' or '/'
    m = _AUTHOR_SPLIT_RE.search(s)
    return s[: m.start()] if m else s


def filter_merged_np_dict(
    merged_ordered: dict,
    min_count: int = 20,
    min_models: int = 10,
    min_authors: int = 5,
    model_id_field: str = "model_id",
) -> OrderedDict:
    """
    Apply thresholds AFTER canonicalization (i.e., after process_np_dict()).

    Requirements:
      - count >= min_count
      - unique model_ids >= min_models
      - unique authors >= min_authors, where author is derived from model_id

    Returns a re-indexed OrderedDict with numeric string keys ("0","1",...)
    preserving original entry data.
    """
    if not merged_ordered:
        return OrderedDict()

    kept_items = []

    for _, entry in merged_ordered.items():
        count = int(entry.get("count", 0) or 0)

        model_ids = entry.get(model_id_field, []) or []
        if not isinstance(model_ids, list):
            model_ids = [model_ids]

        # normalize + unique while preserving order
        seen_m = set()
        model_ids_unique = []
        for mid in model_ids:
            mid_s = str(mid).strip()
            if not mid_s or mid_s in seen_m:
                continue
            seen_m.add(mid_s)
            model_ids_unique.append(mid_s)

        authors = []
        seen_a = set()
        for mid in model_ids_unique:
            a = model_id_to_author(mid)
            if a and a not in seen_a:
                seen_a.add(a)
                authors.append(a)

        if count < min_count:
            continue
        if len(model_ids_unique) < min_models:
            continue
        if len(authors) < min_authors:
            continue

        # optionally store derived authors for downstream use
        entry = dict(entry)
        entry["author"] = authors
        entry["#unique_model_ids"] = len(model_ids_unique)
        entry["#unique_authors"] = len(authors)

        kept_items.append(entry)

    # Repack to numeric keys, keeping a stable sort (by count desc, then phrase)
    kept_items.sort(key=lambda e: (-int(e.get("count", 0)), str(e.get("noun_phrase", ""))))

    out = OrderedDict()
    for i, entry in enumerate(kept_items):
        out[str(i)] = entry
    return out

# ----------------------------
# Usage
# ----------------------------
if __name__ == "__main__":


    with open("2-NP_EXTRACTION/NP_global_dictionary_comfy.json", "r", encoding="utf-8") as f:
        np_dict = json.load(f)

    merged, phrases = process_np_dict(
        np_dict,
        merged_json_path="2-NP_EXTRACTION/NP_comfy_P.json",
        phrase_list_json_path="2-NP_EXTRACTION/NP_compfy_P_llm_go.json",
    )
    print(len(np_dict), "->" )
    print(f"Merged {len(merged)} noun phrases.")
