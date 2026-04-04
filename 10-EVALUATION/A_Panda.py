import json
import re
from typing import Any, Dict

import json
import re
from typing import List, Any, Dict

def parse_llm_json_flex(text: str) -> List[str]:
    s = text.strip()

    parsed: Dict[str, Any] | None = None

    # 1) Try plain JSON directly
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2) Try to extract from ```json ... ``` or ``` ... ```
    if parsed is None:
        fence_match = re.search(
            r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE
        )
        if fence_match:
            inner = fence_match.group(1).strip()
            try:
                parsed = json.loads(inner)
            except json.JSONDecodeError:
                pass

    # 3) Try substring between first '{' and last '}'
    if parsed is None:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = s[start : end + 1]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # 4) Give up if nothing parsed
    if parsed is None:
        snippet = s[:300].replace("\n", "\\n")
        raise ValueError(f"Could not parse JSON from LLM output. Snippet: {snippet}")

    # 5) Normalize output → List[str]
    if isinstance(parsed, dict):
        # Common case: {"features": [...]}
        for value in parsed.values():
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                return value

    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
        return parsed

    raise ValueError(f"Parsed JSON but could not extract List[str]: {parsed}")

import json

import json
import ast

def fragment_to_dict(raw: str) -> dict:
    """
    Robustly convert:
    - JSON fragments: "11": {...}
    - Full JSON: {"11": {...}}
    - Python dict strings (single quotes)
    """
    s = raw.strip()

    # Case 1: already valid JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Case 2: fragment → wrap
    try:
        if s.endswith(","):
            s = s[:-1].rstrip()
        wrapped = "{" + s + "}"
        return json.loads(wrapped)
    except json.JSONDecodeError:
        pass

    # Case 3: Python literal dict (single quotes)
    try:
        return ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Unparseable fragment:\n{s}") from e



import pandas as pd
import ast
import re
import random

# -----------------------
# CONFIG: set these
# -----------------------
CSV_IN = "10-EVALUATION/model_ffs_new.csv" # Input CSV path
CSV_OUT = "10-EVALUATION/model_ffs_eval_T2.csv"

with open(f"10-EVALUATION/llm_ffs/output_A2.json", "r", encoding="utf-8") as f: #A2
    sample_dict = json.load(f)
model_db = sample_dict   # <-- replace this

# -----------------------
# Helpers
# -----------------------
brace_pat = re.compile(r"\{([^{}]+)\}")


def clean_feature_string(s: str) -> str:
    """If string contains {...}, replace the whole string by a random choice of items inside braces.
       Also remove tokens like '(_)' and '(-)'.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return s

    s = str(s)

    # Remove things like: (_)(-)
    s = s.replace("(_)", "").replace("(-)", "")

    # If it has braces anywhere, replace with a random item inside the braces
    m = brace_pat.search(s)
    if m:
        inside = m.group(1)

        # inside typically looks like:  "'Selective Language Modeling', 'Small Language Model'"
        # We'll extract quoted items if possible; fallback to comma-splitting.
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", inside)
        items = [a or b for (a, b) in quoted if (a or b)]
        if not items:
            items = [x.strip() for x in inside.split(",") if x.strip()]

        if items:
            s = random.choice(items)

    # Final whitespace cleanup
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_and_clean_features(cell):
    """Parse Features cell which should be a Python-like list of strings; return cleaned list."""
    if pd.isna(cell):
        return []

    # If it is already a list, keep it; else parse
    if isinstance(cell, list):
        features = cell
    else:
        txt = str(cell).strip()
        try:
            features = ast.literal_eval(txt)
            if not isinstance(features, list):
                features = [txt]
        except Exception:
            # If it isn't valid Python literal list, treat as single string
            features = [txt]

    cleaned = [clean_feature_string(x) for x in features]
    # Remove empties
    cleaned = [x for x in cleaned if x is not None and str(x).strip() != ""]
    return cleaned

def to_list(x):
    """Ensure x is a list of strings."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    return [x]

# -----------------------
# Load CSV
# -----------------------
df = pd.read_csv(CSV_IN)

# -----------------------
# Build model_id as FIRST column
# -----------------------
if 'ModelID' not in df.columns:
    print("Column dont exist")

    df["ModelID"] = df.apply(
        lambda r: f"{r.get('Author')}/{r.get('Model Name')}",
        axis=1
    )
    # move to first column
    df.insert(0, "ModelID", df.pop("ModelID"))

# -----------------------
# Clean Features (list of strings)
# -----------------------
if "Features" in df.columns:
    df["Features"] = df["Features"].apply(parse_and_clean_features)
else:
    df["Features"] = [[] for _ in range(len(df))]


# -----------------------
# Filter to only model_ids in model_db
# -----------------------

before = len(df)

keep_ids = {
    model["model_id"]
    for model in model_db.values()
    if "model_id" in model
}

csv_ids = set(df["ModelID"].astype(str))
missing = sorted(keep_ids - csv_ids)

print("Missing (in model_db but not in CSV):", len(missing))
for mid in missing:
    print("DB:", repr(mid), "len=", len(mid))


df = df[df["ModelID"].isin(keep_ids)].copy()
after = len(df)


print("=== Filter report ===")
print(f"Rows before: {before}")
print(f"Rows after : {after}")
print(f"Kept       : {after} ({after/before:.1%})" if before else "Kept: n/a (no rows)")

# -----------------------
# Add truth + #truth from model_db[model_id]['output']
# -----------------------
def get_truth(model_id):
    for rec in model_db.values():
        if rec.get("model_id") == model_id:
            raw_output = rec.get("output", "")
            shaped_output = parse_llm_json_flex(raw_output)
            # shaped_raw = fragment_to_dict(raw_output)
            # shaped_output = shaped_raw["features"]

            return to_list(shaped_output)
    return []


df["truth"] = df["ModelID"].apply(get_truth)
df["#truth"] = df["truth"].apply(lambda x: len(x) if isinstance(x, list) else 0)

df = df[[col for col in df.columns if col != "Descriptions"] + ["Descriptions"]]

# -----------------------
# Save
# -----------------------
df.to_csv(CSV_OUT, index=False)
print(f"\nSaved: {CSV_OUT}")

unique_features = set(item for sublist in df["Features"] for item in sublist)
features_list = [item for sublist in df["Features"] for item in sublist]

# print results
print("Number of total items:",len(features_list))

print("Number of unique items:", len(unique_features))
