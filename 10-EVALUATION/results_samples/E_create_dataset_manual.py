import os
import json
import pandas as pd

input_folder = "HF-Models-T7"
output_file = "10-EVALUATION/llm_ffs/manuel_gemini_norm.csv"
allowed_path = "10-EVALUATION/llm_ffs/sampled_modelID_glob.json"

with open(allowed_path, "r", encoding="utf-8") as f:
            allowed = json.load(f)
            
import re
from typing import List


# --- Regexes ---
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_TICKS_RE = re.compile(r"`([^`]+)`")  # capture content
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")

STACKTRACE_HINT_RE = re.compile(
    r"^\s*(Traceback\b|File \"|at \w+\.|Exception\b|ERROR\b|WARN(?:ING)?\b|INFO\b)\b",
    re.IGNORECASE
)

# Table-ish patterns are tricky; keep only the very obvious ones
TABLE_PIPE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_RULE_LINE_RE = re.compile(r"^\s*[-+|]{3,}\s*$")

JSONISH_RE = re.compile(r"^\s*[\{\[]")
YAMLY_RE = re.compile(r"^\s*[\w\-]+\s*:\s+.+")  # key: value
SQLISH_RE = re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|WITH)\b", re.IGNORECASE)

NUM_TOKEN_RE = re.compile(r"\b\d[\d.,%]*\b")
WORD_RE = re.compile(r"[A-Za-z]")


import re

HF_MODEL_URL_RE = re.compile(
    r"https?://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:\S*)?"
)

GENERIC_URL_RE = re.compile(r"https?://\S+|www\.\S+")

def replace_urls_preserving_hf(text: str, generic_token: str = " [URL] ") -> str:
    t = text or ""

    # Preserve HF repo slug as a meaningful token
    t = HF_MODEL_URL_RE.sub(lambda m: f" HF_REPO:{m.group(1)} ", t)

    # Replace remaining URLs with a generic token
    t = GENERIC_URL_RE.sub(generic_token, t)

    return t

def extract_code_blocks(text: str) -> List[str]:
    return CODE_FENCE_RE.findall(text or "")


def handle_code_blocks(text: str, mode: str = "placeholder") -> str:
    """
    mode:
      - "keep": keep code blocks as-is
      - "placeholder": replace each code fence with a token
      - "strip": remove code fences entirely (old behavior)
    """
    t = text or ""
    if mode == "keep":
        return t
    if mode == "strip":
        return CODE_FENCE_RE.sub(" ", t)
    # placeholder
    return CODE_FENCE_RE.sub(" [CODE_BLOCK] ", t)


def normalize_whitespace(text: str) -> str:
    t = (text or "").replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def add_soft_newlines(text: str) -> str:
    t = text or ""
    t = re.sub(r"\s+(#{1,6})\s+", r"\n\1 ", t)  # headings
    t = re.sub(r"\s+-\s+", "\n- ", t)          # bullets
    return t


def join_soft_newlines(text: str) -> str:
    lines = (text or "").splitlines()
    out, buf = [], []

    def flush():
        nonlocal buf
        if buf:
            out.append(" ".join(s.strip() for s in buf if s.strip()))
            buf = []

    for line in lines:
        if not line.strip():
            flush()
            out.append("")
        else:
            buf.append(line)

    flush()
    return "\n\n".join([p for p in out if p is not None]).strip()


def is_mostly_numeric(line: str, min_num_tokens: int = 10, max_word_chars: int = 6) -> bool:
    """
    Conservative check: drop only if it looks like a numeric dump.
    Example: long lists of numbers with almost no letters.
    """
    nums = NUM_TOKEN_RE.findall(line)
    if len(nums) < min_num_tokens:
        return False
    letters = WORD_RE.findall(line)
    return len(letters) <= max_word_chars


def looks_like_big_struct_blob(line: str, min_len: int = 220) -> bool:
    """
    Drop only very long single-line JSON/YAML-ish blobs.
    """
    s = line.strip()
    if len(s) < min_len:
        return False
    if JSONISH_RE.match(s):
        return True
    if YAMLY_RE.match(s):
        return True
    return False


def remove_noisy_lines_conservative(text: str) -> str:
    out_lines = []
    for raw in (text or "").splitlines():
        s = raw.strip()

        if not s:
            out_lines.append("")
            continue

        # clear stack traces / log noise
        if STACKTRACE_HINT_RE.match(s):
            continue
        if SQLISH_RE.match(s) and len(s) > 80:
            continue

        # obvious table border lines
        if TABLE_RULE_LINE_RE.match(s):
            continue

        # pure pipe tables: drop only if line is basically pipes + separators + spaces
        if TABLE_PIPE_LINE_RE.match(s) and len(re.sub(r"[|\-:\s]", "", s)) == 0:
            continue

        # numeric dumps (but keep numeric specs that also contain text)
        if is_mostly_numeric(s):
            continue

        # big one-line json/yaml blobs (but don't drop short config-y lines)
        if looks_like_big_struct_blob(s):
            continue

        out_lines.append(raw)

    return "\n".join(out_lines)


def preprocess_text(
    text: str,
    code_blocks: str = "placeholder",  # "keep" | "placeholder" | "strip"
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_html: bool = True,
    inline_code: str = "keep_content",  # "keep_content" | "strip"
    drop_noisy_lines: bool = True,
    url_token: str = " [URL] ",
    email_token: str = " [EMAIL] ",
) -> str:
    """
    Less aggressive cleaning for NP extraction.
    Keeps technical tokens (model names, dtypes, API identifiers) much better.
    """
    cleaned = handle_code_blocks(text, mode=code_blocks)
    cleaned = add_soft_newlines(cleaned)

    # Inline code: keep content, remove just the backticks (recommended)
    if inline_code == "strip":
        cleaned = re.sub(r"`[^`]+`", " ", cleaned)
    else:
        cleaned = INLINE_CODE_TICKS_RE.sub(r"\1", cleaned)

    if remove_html:
        cleaned = HTML_TAG_RE.sub(" ", cleaned)

    # Replace rather than delete: avoids collapsing sentences into garbage
    if remove_urls:
        cleaned = replace_urls_preserving_hf(cleaned, generic_token=" [URL] ")
    if remove_emails:
        cleaned = EMAIL_RE.sub(email_token, cleaned)

    if drop_noisy_lines:
        tmp = add_soft_newlines(cleaned)
        tmp = remove_noisy_lines_conservative(tmp)
        cleaned = join_soft_newlines(tmp)

    cleaned = normalize_whitespace(cleaned)
    return cleaned



text ="RobIt **RobIt** is a RoBERTa-base model for Italian. It has been trained from scratch on the Italian portion of the OSCAR dataset using Flax, including training scripts. This is part of the Flax/Jax Community Week, organised by HuggingFace and TPU usage sponsored by Google. ## Team members - Prateek Agrawal (prateekagrawal) - Tanay Mehta (yotanay) - Shreya Gupta (Sheyz-max) - Ruchi Bhatia (ruchi798) ## Dataset : OSCAR - config : **unshuffled_deduplicated_it** - Size of downloaded dataset files: **26637.62 MB** - Size of the generated dataset: **70661.48 MB** - Total amount of disk used: **97299.10 MB** ## Useful links - Community Week timeline - Community Week README - Community Week thread - Community Week channel - Masked Language Modelling example scripts - Model Repository"

print(preprocess_text(text))
rows = []

for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            rows.append(data)

# Create DataFrame
df = pd.DataFrame(rows)

# Filter + select columns
filtered_df = df[df["modelID"].isin(allowed)][["modelID", "description"]]
filtered_df["normalized"] = filtered_df["description"].apply(preprocess_text)
filtered_df["truth"] = pd.NA
# Save to CSV
filtered_df.to_csv(output_file, index=False)

print("Filtered CSV created.")
