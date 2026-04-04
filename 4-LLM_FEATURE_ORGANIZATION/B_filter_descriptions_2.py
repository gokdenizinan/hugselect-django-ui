import json
import re

DESC_CUT_RE = re.compile(r"\bthe description is\b.*", re.IGNORECASE | re.DOTALL)

LABEL_ABBR_RE = re.compile(
    r"""(?P<prefix>.*?)
        (?P<label>[^/()]+?)      # label segment (no slash/paren), non-greedy
        \s*\(\s*(?P<abbr>[A-Za-z0-9][A-Za-z0-9\-/]*)\s*\)
    """,
    re.VERBOSE,
)

UNDERSCORE_RE = re.compile(r"^\s*(?P<abbr>[A-Za-z0-9][A-Za-z0-9\-/]*)\s*\(_\)")

def parse_output_relevant_abbreviation(output_val):
    try:
        if isinstance(output_val, str):
            data = json.loads(output_val)
            return data.get("relevant_abbreviation") or data.get("relevant_expansion")
        if isinstance(output_val, dict):
            return output_val.get("relevant_abbreviation") or output_val.get("relevant_expansion")
    except Exception:
        return None
    return None

def extract_abbr_from_text(text: str):
    """
    Returns abbreviation string if found, else None.
    Handles:
      - 'ABBR(_)...'
      - '... (ABBR) ...'
    """
    if not text:
        return None

    m1 = UNDERSCORE_RE.search(text)
    if m1:
        return m1.group("abbr")

    m2 = LABEL_ABBR_RE.search(text)
    if m2:
        return m2.group("abbr")

    return None

def solved_abbr_set(existing_solved: dict) -> set[str]:
    """
    existing_solved is your old 'abbreviation_llm_GG output' dict:
      { "280": { "abreviation": "...", "output": "..."} , ... }
    We treat an abbreviation as solved if:
      - it has a parsed output relevant_abbreviation (i.e., model answered)
      - AND we can extract the abbreviation token from the 'abreviation' field
    """
    solved = set()
    for _, item in existing_solved.items():
        text = item.get("abreviation", "") or ""
        out = parse_output_relevant_abbreviation(item.get("output"))
        if not out:
            continue

        abbr = extract_abbr_from_text(text)
        if abbr:
            solved.add(abbr)

    return solved

def filter_new_llm_input(new_llm_input: dict, solved_abbrs: set[str]):
    """
    new_llm_input looks like:
      { "abbreviation_0": "<string>", "abbreviation_1": "<string>", ... }
    Returns filtered dict (same format) + some stats.
    """
    kept = {}
    removed = {}

    for k, v in new_llm_input.items():
        # v can be string, or sometimes dict—handle both
        text = v if isinstance(v, str) else (v.get("abreviation") or v.get("abbreviation") or v.get("text") or "")
        abbr = extract_abbr_from_text(text)

        if abbr and abbr in solved_abbrs:
            removed[k] = v
        else:
            kept[k] = v

    return kept, removed

# --------- paths ---------
OLD_SOLVED_PATH_1 = "4-LLM_FEATURE_ORGANIZATION/output_A1.json"   # <-- your existing solved outputs
OLD_SOLVED_PATH_2 = "4-LLM_FEATURE_ORGANIZATION/output_org_A2.json"   # <-- your existing solved outputs
OLD_SOLVED_PATH_3 = "4-LLM_FEATURE_ORGANIZATION/output_org_A2.json"   # <-- your existing solved outputs

NEW_INPUT_PATH  = "4-LLM_FEATURE_ORGANIZATION/abbreviation_to_llm_GG.json"      # <-- the new LLM input you want to shrink

OUT_FILTERED_PATH = "4-LLM_FEATURE_ORGANIZATION/abbreviation_to_llm_GG_filtered.json"
OUT_REMOVED_PATH  = "4-LLM_FEATURE_ORGANIZATION/abbreviation_to_llm_GG_removed.json"

# --------- run ---------
with open(OLD_SOLVED_PATH_1, "r", encoding="utf-8") as f:
    old_solved1 = json.load(f)

with open(OLD_SOLVED_PATH_2, "r", encoding="utf-8") as f:
    old_solved2 = json.load(f)

with open(OLD_SOLVED_PATH_3, "r", encoding="utf-8") as f:
    old_solved3 = json.load(f)

# Combine all solved sets
old_solved = {}
old_solved.update(old_solved1)
old_solved.update(old_solved2)
old_solved.update(old_solved3)  

with open(NEW_INPUT_PATH, "r", encoding="utf-8") as f:
    new_input = json.load(f)

solved = solved_abbr_set(old_solved)
kept, removed = filter_new_llm_input(new_input, solved)

with open(OUT_FILTERED_PATH, "w", encoding="utf-8") as f:
    json.dump(kept, f, indent=2, ensure_ascii=False)

with open(OUT_REMOVED_PATH, "w", encoding="utf-8") as f:
    json.dump(removed, f, indent=2, ensure_ascii=False)

print("Solved abbreviations found:", len(solved))
print("Original new input size:", len(new_input))
print("Kept (still need LLM):", len(kept))
print("Removed (already solved):", len(removed))
print("Wrote:", OUT_FILTERED_PATH)
print("Wrote removed log:", OUT_REMOVED_PATH)
