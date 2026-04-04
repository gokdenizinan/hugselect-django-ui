import pandas as pd
import json
import ast

ORIGINAL_PATH = "12-EVALUATION_QUAL/sentiment_sample_A.csv"
OUT_PATH      = "12-EVALUATION_QUAL/sentiment_sample_B.csv"

orig = pd.read_csv(ORIGINAL_PATH, sep=";", quotechar='"', engine="python")
import pandas as pd
import json
import ast
import re

def extract_sentiment_cell(x):
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
        if isinstance(obj, dict) and "sentiment" in obj:
            return obj["sentiment"]
    except Exception:
        pass

    # Try Python literal (handles single quotes)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict) and "sentiment" in obj:
            return obj["sentiment"]
    except Exception:
        pass

    # Last-resort regex (handles slightly broken JSON)
    m = re.search(r'"sentiment"\s*:\s*"([^"]+)"', s)
    if m:
        return m.group(1)

    return pd.NA

# Parse feature columns
df = orig.copy()
df["chatgpt"] = df["chatgpt"].apply(extract_sentiment_cell)
df["gemini"]  = df["gemini"].apply(extract_sentiment_cell)

# Map mode -> predicted
mapping = {-1: "unclear", 0: "negative", 1: "neutral", 2: "positive"}

# choose the right source column here:
df["predicted"] = df["mode_2"].map(mapping)

df = df.rename(columns={
    "model_id": "ModelID",
    "mode": "mode_2",
    "reviews": "Review_Processed",
    "original": "Review_Original"
})


# Select output columns (include counts if you want them)
out_cols = ["ModelID", "Review_Processed", "sent_1", "sent_2", "sent_3", "mode_2",
            "chatgpt", "gemini", "predicted"]
out_cols = [c for c in out_cols if c in df.columns]

df = df[out_cols]

df.to_csv(OUT_PATH, sep=";", index=False)
print(f"Wrote: {OUT_PATH}")