import pandas as pd
import json
import ast
# A00 A YAZ SONUCLARI
# A00 DAN AA00 OLUYOR KAYDEDINCE
ORIGINAL_PATH = "12-EVALUATION_QUAL/quality_sample_AA00.csv"
OUT_PATH      = "12-EVALUATION_QUAL/quality_sample_B.csv"

orig = pd.read_csv(ORIGINAL_PATH, sep=";", quotechar='"', engine="python")
import pandas as pd
import json
import ast
import re

def extract_primary_cell(x):
    """
    Takes a cell containing a JSON/dict-like string such as:
      '{"sentiment":"neutral","confidence":0.84,"reason":"..."}'
    and returns the sentiment string, e.g. "neutral".
    Returns NaN if it can't be parsed.
    """
    if pd.isna(x):
        return pd.NA

    # Sometimes cells contain odd whitespace/control chars
    s = str(x).strip()
    if not s:
        return pd.NA

    # Remove BOM/zero-width chars that can break json.loads
    s = s.replace("\ufeff", "").replace("\u200b", "")

    # Strip one layer of wrapping quotes if present
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # Try JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "Primary_Category" in obj:
            return obj["Primary_Category"]
    except Exception:
        pass

    # Try Python literal (handles single quotes)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict) and "Primary_Category" in obj:
            return obj["Primary_Category"]
    except Exception:
        pass

    # Last-resort regex (handles slightly broken JSON)
    m = re.search(r'"Primary_Category"\s*:\s*"([^"]+)"', s)
    if m:
        return m.group(1)

    return pd.NA


def extract_secondary_cell(x):
    """
    Takes a cell containing a JSON/dict-like string such as:
      '{"sentiment":"neutral","confidence":0.84,"reason":"..."}'
    and returns the sentiment string, e.g. "neutral".
    Returns NaN if it can't be parsed.
    """
    if pd.isna(x):
        return pd.NA

    # Sometimes cells contain odd whitespace/control chars
    s = str(x).strip()
    if not s:
        return pd.NA

    # Remove BOM/zero-width chars that can break json.loads
    s = s.replace("\ufeff", "").replace("\u200b", "")

    # Strip one layer of wrapping quotes if present
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # Try JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "Secondary_Categories" in obj:
            return obj["Secondary_Categories"]
    except Exception:
        pass

    # Try Python literal (handles single quotes)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict) and "Secondary_Categories" in obj:
            return obj["Secondary_Categories"]
    except Exception:
        pass

    # Last-resort regex (handles slightly broken JSON)
    m = re.search(r'"Secondary_Categories"\s*:\s*"([^"]+)"', s)
    if m:
        return m.group(1)

    return pd.NA

# Parse feature columns
df = orig.copy()
df["chatgpt_primary"] = df["chatgpt"].apply(extract_primary_cell)
df["gemini_primary"]  = df["Gemini"].apply(extract_primary_cell)
df["chatgpt_secondary"] = df["chatgpt"].apply(extract_secondary_cell)
df["gemini_secondary"]  = df["Gemini"].apply(extract_secondary_cell)


df = df.rename(columns={
    "model_id": "ModelID",
    "Primary_Category": "predicted",
    "reviews": "Review_Processed",
    "original": "Review_Original"
})


# Select output columns (include counts if you want them)
out_cols = ["ModelID", "Review_Processed",
            "chatgpt_primary", "gemini_primary", "chatgpt_secondary", "gemini_secondary", "predicted"]
out_cols = [c for c in out_cols if c in df.columns]

df = df[out_cols]

df.to_csv(OUT_PATH, sep=";", index=False)
print(f"Wrote: {OUT_PATH}")