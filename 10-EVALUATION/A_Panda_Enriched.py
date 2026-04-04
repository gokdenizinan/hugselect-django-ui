import pandas as pd
import numpy as np
import json
import ast

# ---- paths (edit these) ----
ORIGINAL_PATH = "10-EVALUATION/llm_ffs/manuel_gemini_chat_3.csv"        # your "original" export
NEW_PATH      = "10-EVALUATION/models_ffs/model_ffs_eval_T5.csv"         # your "new" file
OUT_PATH      = "10-EVALUATION/enriched_ffs/enriched_ffs_eval_T5.csv"

# ---- read ----
orig = pd.read_csv(ORIGINAL_PATH,sep=";",quotechar='"',engine="python")  # more tolerant than C engine

new = pd.read_csv(NEW_PATH)

# ---- normalize join keys ----
orig["modelID"] = orig["modelID"].astype(str).str.strip()
new["ModelID"]  = new["ModelID"].astype(str).str.strip()

import json

def extract_features(x):
    if pd.isna(x):
        return []
    try:
        return json.loads(x)["features"]
    except Exception:
        return []

orig["chat_truth"] = orig["chat_truth"].apply(extract_features)
orig["gemini_truth"] = orig["gemini_truth"].apply(extract_features)

# ---- pull needed columns from new and rename ----
# Features  -> Pipeline_Features
# truth     -> Instruct_Features
new_keep = new[["ModelID", "Features", "truth", "Descriptions"]].copy()
new_keep = new_keep.rename(
    columns={
        "Features": "Pipeline_Features",
        "truth": "Instruct_Features",
    }
)

# ---- left merge into original ----
df = orig.merge(new_keep, how="left", left_on="modelID", right_on="ModelID")
df = df.drop(columns=["ModelID"], errors="ignore")


# ---- feature parser: dict-string -> list ----
def parse_features_cell(x):
    """
    Converts values like:
      '{"features": ["image generation", "diffusers library"]}'
    into:
      ["image generation", "diffusers library"]

    Also handles already-list, dict, NaN, messy quoting.
    """
    if pd.isna(x):
        return []

    # If already list
    if isinstance(x, list):
        return [str(i) for i in x if pd.notna(i)]

    # If already dict
    if isinstance(x, dict):
        v = x.get("features", [])
        if isinstance(v, list):
            return [str(i) for i in v if pd.notna(i)]
        return []

    # Must be string-ish
    s = str(x).strip()
    if not s:
        return []

    # Some CSVs contain quoted JSON like: "\"{...}\""
    # Strip one layer of surrounding quotes if present
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # Try JSON first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "features" in obj and isinstance(obj["features"], list):
            return [str(i) for i in obj["features"] if pd.notna(i)]
        if isinstance(obj, list):
            return [str(i) for i in obj if pd.notna(i)]
    except Exception:
        pass

    # Fallback: Python literal eval (handles single quotes, etc.)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict) and "features" in obj and isinstance(obj["features"], list):
            return [str(i) for i in obj["features"] if pd.notna(i)]
        if isinstance(obj, list):
            return [str(i) for i in obj if pd.notna(i)]
    except Exception:
        pass

    # Last resort: can't parse -> empty list
    return []

FEATURE_COLS = ["Pipeline_Features", "Instruct_Features", "Gemini_Features", "ChatGPT_Features"]
for c in FEATURE_COLS:
    if c in df.columns:
        df[c] = df[c].apply(parse_features_cell)

# ---- add count columns ----
for c in FEATURE_COLS:
    if c in df.columns:
        df[f"{c[:4]}_count"] = df[c].apply(len)

# ---- reorder: pipeline/instruct before gemini/chatgpt ----
# Keep relative ordering of everything else
cols = list(df.columns)

# Move "description" -> end and rename to Description_norm later
desc_col = "description"
has_desc = desc_col in cols

# Build desired feature block order (only those that exist)
feature_block = [c for c in ["Pipeline_Features", "Instruct_Features", "Gemini_Features", "ChatGPT_Features"] if c in cols]
count_block   = [f"{c}_count" for c in feature_block if f"{c}_count" in cols]

# Remove these from current columns so we can reinsert cleanly
to_remove = set(feature_block + count_block)
base_cols = [c for c in cols if c not in to_remove and c != desc_col]

# Insert feature_block + count_block right before the first of Gemini/ChatGPT if present,
# otherwise just after modelID if present, otherwise at the front.
insert_after = None
for anchor in ["Gemini_Features", "ChatGPT_Features"]:
    if anchor in cols:
        insert_after = anchor
        break

if insert_after and insert_after in base_cols:
    idx = base_cols.index(insert_after)
    # Insert before anchor
    new_cols = base_cols[:idx] + feature_block + count_block + base_cols[idx:]
else:
    if "modelID" in base_cols:
        idx = base_cols.index("modelID") + 1
        new_cols = base_cols[:idx] + feature_block + count_block + base_cols[idx:]
    else:
        new_cols = feature_block + count_block + base_cols

# Add description at the end (renamed)
if has_desc:
    df = df.rename(columns={desc_col: "Description_norm"})
    new_cols = [c for c in new_cols if c != desc_col] + ["Description_norm"]

# Ensure Descriptions from new is present (already merged); keep wherever it currently is,
# unless you want it at the end too. (Leaving it in-place as requested.)
df = df[new_cols + [c for c in df.columns if c not in new_cols]]
df = df.rename(
    columns={
        "modelID": "ModelID",  # just to be explicit
        "chat_truth": "ChatGPT_Features",
        "gemini_truth": "Gemini_Features",
    }
)
for c in ["Gemini_Features", "ChatGPT_Features"]:
    if c in df.columns:
        df[f"{c[:4]}_count"] = df[c].apply(len)


df = df[["ModelID", "Pipeline_Features", "Pipe_count", "Instruct_Features", "Inst_count", "ChatGPT_Features", "Chat_count", "Gemini_Features", "Gemi_count", "Descriptions", "Description_norm"]]
# ---- write ----
df.to_csv(OUT_PATH, index=False)
print(f"Wrote: {OUT_PATH}")