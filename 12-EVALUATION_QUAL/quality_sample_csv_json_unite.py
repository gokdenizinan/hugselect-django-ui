import json
import pandas as pd

# Input files
csv_file = "12-EVALUATION_QUAL/quality_sample_C.csv"
gemini_json_file = "12-EVALUATION_QUAL/quality_sample_C_gemini.json"
chat_json_file = "12-EVALUATION_QUAL/quality_sample_C_chat.json"

# Output file
output_csv_file = "12-EVALUATION_QUAL/quality_sample_Y.csv"

df = pd.read_csv(csv_file, sep=";", engine="python")

with open(gemini_json_file, "r", encoding="utf-8") as f:
    gemini_data = json.load(f)

with open(chat_json_file, "r", encoding="utf-8") as f:
    chat_data = json.load(f)

# Ensure columns exist (don’t overwrite if already present)
for col in ["gemini_primary", "gemini_secondary", "chatgpt_primary", "chatgpt_secondary"]:
    if col not in df.columns:
        df[col] = ""

# Helper function: only fill if empty
def safe_set(df, idx, col, value):
    if idx in df.index:
        if pd.isna(df.at[idx, col]) or df.at[idx, col] == "":
            df.at[idx, col] = value

# Fill Gemini (shifted up by 1 row)
for key, value in gemini_data.items():
    row_idx = int(key) - 1  # ← shift up

    safe_set(df, row_idx, "gemini_primary",
             value.get("Primary_Category", ""))

    safe_set(df, row_idx, "gemini_secondary",
             json.dumps(value.get("Secondary_Categories", []), ensure_ascii=False))

# Fill ChatGPT (shifted up by 1 row)
for key, value in chat_data.items():
    row_idx = int(key) - 1  # ← shift up

    safe_set(df, row_idx, "chatgpt_primary",
             value.get("Primary_Category", ""))

    safe_set(df, row_idx, "chatgpt_secondary",
             json.dumps(value.get("Secondary_Categories", []), ensure_ascii=False))

# Save
df.to_csv(output_csv_file, sep=";", index=False, encoding="utf-8-sig")

print(f"Saved updated CSV as: {output_csv_file}")