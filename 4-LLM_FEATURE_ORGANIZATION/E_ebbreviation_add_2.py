import json

with open("4-LLM_FEATURE_ORGANIZATION/NP_GG_processed.json", "r", encoding="utf-8") as f:
    dict_1 = json.load(f)

with open("4-LLM_FEATURE_ORGANIZATION/output_suan_G0.json", "r", encoding="utf-8") as f:
    dict_2 = json.load(f)

print(len(dict_1))
import json
import re

DESC_CUT_RE = re.compile(r"\bthe description is\b.*", re.IGNORECASE | re.DOTALL)

# Captures: "<label> (ABBR)" where label is the text immediately before the parentheses.
# We only grab the final label segment (after the last slash) to handle "A/B (SER)" -> label="B"
LABEL_ABBR_RE = re.compile(
    r"""(?P<prefix>.*?)
        (?P<label>[^/()]+?)      # label segment (no slash/paren), non-greedy
        \s*\(\s*(?P<abbr>[A-Za-z0-9][A-Za-z0-9\-/]*)\s*\)
    """,
    re.VERBOSE,
)

# Captures: "ABBR(_)" at the beginning
UNDERSCORE_RE = re.compile(r"^\s*(?P<abbr>[A-Za-z0-9][A-Za-z0-9\-/]*)\s*\(_\)")

def strip_description(text: str) -> str:
    return DESC_CUT_RE.sub("", text or "").strip()

def parse_output_relevant_abbreviation(output_val):
    try:
        if isinstance(output_val, str):
            data = json.loads(output_val)

            value = data.get("relevant_abbreviation") or data.get("relevant_expansion")
            return value
        if isinstance(output_val, dict):
            value = output_val.get("relevant_abbreviation") or output_val.get("relevant_expansion")
            return value
    except Exception:
        return None
    return None

def build_lookup(abbrev_openings: dict) -> dict[str, str]:
    """
    Map abbreviation -> relevant_abbreviation, e.g. "SER" -> "Speech Emotion Recognition"
    """
    lookup = {}
    for _, item in abbrev_openings.items():
        text = item.get("abreviation", "") or ""
        out = parse_output_relevant_abbreviation(item.get("output"))

        if not out:
            continue

        m1 = UNDERSCORE_RE.search(text)
        if m1:
            lookup[m1.group("abbr")] = out
            continue

        m2 = LABEL_ABBR_RE.search(text)
        if m2:
            lookup[m2.group("abbr")] = out

    return lookup
def keep_through_inserted_abbr(s: str, abbr: str) -> str:
    token = f"({abbr})"
    start = s.find(token)
    if start == -1:
        return s
    return s[: start + len(token)]


def rewrite_noun_phrase(phrase: str, expansion: str, abbr: str) -> str:
    if not phrase:
        return phrase

    repl = f"{expansion} ({abbr})"

    if UNDERSCORE_RE.search(phrase):
        phrase = UNDERSCORE_RE.sub(repl, phrase, count=1)
        phrase = keep_through_inserted_abbr(phrase, abbr)  # <- pass abbr
        return strip_description(phrase)

    phrase = LABEL_ABBR_RE.sub(lambda m: repl, phrase, count=1)
    phrase = keep_through_inserted_abbr(phrase, abbr)      # <- pass abbr
    return strip_description(phrase)


# def rewrite_noun_phrase(phrase: str, expansion: str, abbr: str) -> str:
#     """
#     Replace either:
#       - "ABBR(_)"  -> "EXPANSION (ABBR)"
#       - "<anything> <label> (ABBR)" -> "EXPANSION (ABBR)"
#     Then remove "the description is..." tail.
#     """
#     if not phrase:
#         return phrase

#     # Case 1: ABBR(_)
#     if UNDERSCORE_RE.search(phrase):
#         phrase = UNDERSCORE_RE.sub(f"{expansion} ({abbr})", phrase, count=1)
#         return strip_description(phrase)

#     # Case 2: label (ABBR) somewhere in the text
#     # Replace the FIRST occurrence of "<...>label (ABBR)" with just "EXPANSION (ABBR)"
#     # This cleanly handles "Software Entity Recognition/Speech Emotion Recognition (SER) ..."
#     def _sub(m):
#         return f"{expansion} ({abbr})"

#     phrase = LABEL_ABBR_RE.sub(_sub, phrase, count=1)
#     return strip_description(phrase)

def apply_replacements(noun_phrases: dict, abbrev_openings: dict) -> dict:
    lookup = build_lookup(abbrev_openings)

    for _, rec in noun_phrases.items():
        phrase = rec.get("noun_phrase", "") or ""

        # Find abbreviation in the phrase
        m1 = UNDERSCORE_RE.search(phrase)
        if m1:
            abbr = m1.group("abbr")
        else:
            m2 = LABEL_ABBR_RE.search(phrase)
            abbr = m2.group("abbr") if m2 else None

        if not abbr:
            continue

        expansion = lookup.get(abbr)
        if not expansion:
            continue

        rec["noun_phrase"] = rewrite_noun_phrase(phrase, expansion, abbr)

    return noun_phrases


# ---- Test with your SER example ----
noun_phrases = {
    "NP_384": {
        "noun_phrase": "Software Entity Recognition/Speech Emotion Recognition (SER) the description is: Speech Emotion Recognition (SER) is a task that enables the model to recognize emotions from speech inputs.",
        "count": 9,
        "model_id": ["..."]
    }
}

abbrev_openings = {
    "0": {
        "abreviation": "Software Entity Recognition/Speech Emotion Recognition (SER) the description is: Speech Emotion Recognition (SER) is a task that enables the model to recognize emotions from speech inputs.",
        "output": "{\"relevant_abbreviation\": \"Speech Emotion Recognition\"}",
        "model_used": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    }
}

updated = apply_replacements(dict_1, dict_2)
# print(updated["NP_384"]["noun_phrase"])
# -> "Speech Emotion Recognition (SER)"

with open("4-LLM_FEATURE_ORGANIZATION/NP_GG_fin.json", "w", encoding="utf-8") as f:
    json.dump(updated, f, indent=2, ensure_ascii=False)