import re
import spacy
import unicodedata
from typing import Dict, Any, List, Tuple

# -------------------- config --------------------
DET_LIKE = {"the", "a", "an", "this", "that", "these", "those"}

SEPS = r"(?:\s*[-–—]{1,3}\s*|\s*:\s*|\s*\|\s*)"
GENERIC_RIGHT = {
    "chat models", "models", "model", "type", "problem type",
    "function call", "function response", "benchmarks", "benchmark",
    "tokens", "response", "call", "leaderboard",
}

UNITS_HINT = re.compile(r"(\d|k|m|%|token|char|character|characters|字|万|约|左右|around|approx)", re.I)
GENERIC_TAIL = {"tasks", "task", "pairs", "pair", "models", "model"}
LABEL_LEFT = {"haystack", "needle", "training data", "train data", "data", "type", "category"}

RE_LETTER_DOT_LETTER = re.compile(r"[A-Za-z]\.[A-Za-z]")
RE_ENDS_JPG = re.compile(r"\.jpg$", re.I)

# -------------------- spaCy --------------------
nlp = spacy.load("en_core_web_sm")


# -------------------- helpers --------------------
def _dedupe_adjacent_words(s: str) -> str:
    toks = s.split()
    out = []
    for t in toks:
        if not out or out[-1].lower() != t.lower():
            out.append(t)
    return " ".join(out)


def _balance_parentheses(s: str) -> str:
    if not s:
        return s
    open_ct = s.count("(")
    close_ct = s.count(")")
    if open_ct > close_ct:
        s = s + (")" * (open_ct - close_ct))
    return s


def _ratio_digits(s: str) -> float:
    t = re.sub(r"\s+", "", s or "")
    if not t:
        return 0.0
    digits = sum(ch.isdigit() for ch in t)
    return digits / len(t)


def _ratio_non_english_or_signs(s: str) -> float:
    """
    "bad" chars:
      - non-ASCII alphabetic chars (covers Chinese etc.)
      - Unicode symbols (category starts with 'S') (covers =+ etc. and emojis)
    """
    t = re.sub(r"\s+", "", s or "")
    if not t:
        return 0.0
    bad = 0
    for ch in t:
        cat = unicodedata.category(ch)
        is_symbol = cat.startswith("S")
        is_non_ascii_alpha = ch.isalpha() and ord(ch) > 127
        if is_symbol or is_non_ascii_alpha:
            bad += 1
    return bad / len(t)


def _stable_unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# -------------------- phrase compaction --------------------
def compact_phrase(s: str) -> str:
    if not s:
        return s

    s = _balance_parentheses(s)
    s = s.strip().strip('"').strip("'").strip()

    s = s.replace("：", ":").replace("**", "")
    s = re.sub(r"\s+", " ", s)

    s = re.sub(r"^\s*#+\s*", "", s)
    s = re.sub(r"\s*#+\s*", " ", s).strip()

    def drop_paren(m):
        inner = m.group(1)
        return "" if UNITS_HINT.search(inner) else m.group(0)

    s = re.sub(r"\(([^)]*)\)", drop_paren, s).strip()

    if ":" in s:
        left, right = [p.strip() for p in s.split(":", 1)]
        if left.lower() in LABEL_LEFT and right:
            s = right

    s = _dedupe_adjacent_words(s)
    s = re.sub(r"\s+", " ", s).strip(" -–—:|")
    return s.strip()


def compact_separators(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    parts = re.split(SEPS, t)
    if len(parts) <= 1:
        return t

    left = parts[0].strip()
    right_raw = parts[-1].strip()
    right = right_raw.lower()

    if right in GENERIC_RIGHT or len(right.split()) <= 2:
        return left

    return left if len(left) <= len(right_raw) else right_raw


# -------------------- noun phrase compaction --------------------
def compact_np_span(span) -> str:
    toks = [t for t in span if not t.is_space]

    while toks and (toks[0].is_punct or toks[0].text in {"#", "*", "-", "–", "—"}):
        toks.pop(0)

    # determiners FIRST
    while toks and toks[0].pos_ == "DET" and toks[0].lower_ in DET_LIKE:
        toks.pop(0)

    while toks and toks[-1].is_punct:
        toks.pop()

    return " ".join(t.text for t in toks).strip()


def drop_generic_tail(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    parts = s.split()
    if parts and parts[-1].lower() in GENERIC_TAIL:
        return " ".join(parts[:-1]).strip()
    return s


def compact_np_from_text(text: str) -> str:
    s = _balance_parentheses(text or "")
    s = compact_separators(s)
    s = compact_phrase(s)
    if not s:
        return ""

    doc = nlp(s)
    s = compact_np_span(doc[:])
    s = drop_generic_tail(s)
    s = re.sub(r"\s+", " ", s).strip(" -–—:|")
    return s.strip()


# -------------------- filters --------------------
def should_drop_np(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True

    if s.startswith("/"):
        return True

    if RE_ENDS_JPG.search(s):
        return True

    if RE_LETTER_DOT_LETTER.search(s):
        return True

    if len(s) < 3:
        return True

    if len(s) > 50:
        return True

    if _ratio_digits(s) >= 0.80:
        return True

    if _ratio_non_english_or_signs(s) >= 0.60:
        return True

    return False


# -------------------- merging duplicates --------------------
def merge_entries(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)

    if "count" in a or "count" in b:
        out["count"] = int(a.get("count", 0)) + int(b.get("count", 0))

    if "model_id" in a or "model_id" in b:
        out["model_id"] = _stable_unique(list(a.get("model_id", [])) + list(b.get("model_id", [])))

    if "sentence" in a or "sentence" in b:
        out["sentence"] = _stable_unique(list(a.get("sentence", [])) + list(b.get("sentence", [])))

    if "merged_from" in a or "merged_from" in b:
        out["merged_from"] = _stable_unique(list(a.get("merged_from", [])) + list(b.get("merged_from", [])))

    for k, v in b.items():
        if k not in out:
            out[k] = v

    return out


def _pick_canonical_np(variants: List[str]) -> str:
    """
    Choose which casing to keep when merging case-insensitive duplicates.
    Preference:
      1) Title Case-ish (has at least one uppercase and one lowercase) OR just normal mixed case
      2) otherwise the longest (more informative)
      3) otherwise first seen
    """
    def score(s: str) -> Tuple[int, int]:
        has_upper = any(c.isupper() for c in s)
        has_lower = any(c.islower() for c in s)
        mixed = 1 if (has_upper and has_lower) else 0
        return (mixed, len(s))

    return max(variants, key=score)


# -------------------- main processing --------------------
def filter_compact_and_merge(data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    A) remove keys where len(sentence) <= 1
    B) compact noun_phrase
    C) apply extra filters (drop empties, numeric/non-english/signs rules, etc.)
    D) merge duplicates case-insensitively (NEW)
    E) return new dict with fresh NP_* keys
    """
    cleaned_items: List[Dict[str, Any]] = []
    for _, v in data.items():
        if len(v.get("sentence", [])) <= 1:
            continue

        new_v = dict(v)
        new_np = compact_np_from_text(new_v.get("noun_phrase", ""))

        if should_drop_np(new_np):
            continue

        new_v["noun_phrase"] = new_np
        cleaned_items.append(new_v)

    # D: merge duplicates case-insensitively
    groups: Dict[str, Dict[str, Any]] = {}          # key: np.casefold()
    np_variants: Dict[str, List[str]] = {}          # store seen casing variants for canonical choice

    for v in cleaned_items:
        np_txt = v["noun_phrase"]
        key = np_txt.casefold()  # stronger than lower() for unicode
        if key in groups:
            groups[key] = merge_entries(groups[key], v)
            np_variants[key].append(np_txt)
        else:
            groups[key] = v
            np_variants[key] = [np_txt]

    # choose canonical casing per group and set it
    merged_items: List[Dict[str, Any]] = []
    for key, v in groups.items():
        canon = _pick_canonical_np(np_variants[key])
        vv = dict(v)
        vv["noun_phrase"] = canon
        merged_items.append(vv)

    # E: return with new NP_1... keys (stable order of first appearance)
    out: Dict[str, Dict[str, Any]] = {}
    for i, v in enumerate(merged_items, start=1):
        out[f"NP_{i}"] = v

    return out


# -------------------- example usage --------------------
if __name__ == "__main__":

    import json
    with open("2-NP_EXTRACTION/NP_global_dictionary_suan.json", "r") as f:
        suan = json.load(f)
    
    filtered = filter_compact_and_merge(suan)

print(len(suan))
print(len(filtered))
output_path = "2-NP_EXTRACTION/NP_global_dictionary_suan_filtered_2.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2, ensure_ascii=False)

