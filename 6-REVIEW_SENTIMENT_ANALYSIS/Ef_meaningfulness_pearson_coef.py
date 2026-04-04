import json
import pandas as pd
import json
import ast
import numpy as np
import re

# # Now compute Pearson correlation
# ### CALCULATE PEARSON
# df = pd.read_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/meaningfulness_llm_bert_scores_500.csv", encoding="utf-8")

# def extract_grade(x):
#     try:
#         if isinstance(x, str) and x.strip():  # non-empty string
#             return int(json.loads(x)['grade'])
#         elif isinstance(x, dict):  # already a dictionary
#             return int(x['grade'])
#         else:
#             return np.nan  # for empty or invalid values
#     except (json.JSONDecodeError, KeyError, TypeError):
#         return np.nan

# def extract_meaningfulness(x):
#     try:
#         if isinstance(x, dict) and 'score_0_100' in x:
#             return int(x['score_0_100'])
#         elif isinstance(x, str) and x.strip():
#             return int(json.loads(x)['score_0_100'])
#         else:
#             return np.nan
#     except (json.JSONDecodeError, KeyError, TypeError):
#         return np.nan

# df['output_grade'] = df['output'].apply(extract_grade)
# df['meaningfulness_score'] = df['meaningfulness'].apply(extract_meaningfulness)



# df.to_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/meaningfulness_llm_bert_scores_500v2.csv", index=False, encoding="utf-8")

# # # Pearson correlation
# # pearson_corr = df['output_grade'].corr(df['meaningfulness'], method='pearson')

# # # Spearman correlation
# # spearman_corr = df['output_grade'].corr(df['meaningfulness'], method='spearman')

# # print("Pearson correlation:", pearson_corr)
# # print("Spearman correlation:", spearman_corr)

    

# helper to clean code fences and whitespace
_CODE_FENCE_RE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', flags=re.IGNORECASE)

def _strip_code_fence(s: str) -> str:
    """Remove surrounding triple-backtick fences and surrounding whitespace."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    # remove starting ``` or ```json
    # remove trailing ```
    # using regex substitute only at start and end
    s = re.sub(r'^\s*```(?:json|json\n)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```\s*$', '', s)
    return s.strip()

def parse_to_dict(value):
    """
    Try to parse a value that might be:
      - a dict already
      - a JSON string (double quotes)
      - a Python dict string (single quotes) or other literal -> ast.literal_eval
      - wrapped in ```json ... ```
    Returns a dict on success, None on failure.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    # If it's already a dict-like
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        s = _strip_code_fence(value)
        if s == "":
            return None

        # try json.loads first
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            pass

        # fall back to ast.literal_eval for Python-like dicts
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            pass

        # last resort: try to find a JSON-like substring between first { and last }
        try:
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1 and end > start:
                sub = s[start:end+1]
                try:
                    return json.loads(sub)
                except Exception:
                    try:
                        return ast.literal_eval(sub)
                    except Exception:
                        return None
        except Exception:
            return None

    # unknown type
    return None

def extract_grade_cell(x):
    d = parse_to_dict(x)
    if not d:
        return np.nan
    # grade might be nested or be a string; handle exceptions
    try:
        g = d.get('grade') if isinstance(d, dict) else None
        if g is None:
            # sometimes top-level has other wrappers, try scanning for "grade" anywhere
            if isinstance(d, dict):
                for v in d.values():
                    if isinstance(v, dict) and 'grade' in v:
                        g = v['grade']
                        break
        if g is None:
            return np.nan
        return int(round(float(g)))
    except Exception:
        return np.nan

def extract_meaningfulness_score_cell(x):
    d = parse_to_dict(x)
    if not d:
        return np.nan
    try:
        # prefer top-level 'score_0_100'
        if 'score_0_100' in d:
            return int(round(float(d['score_0_100'])))
        # maybe it's nested under a key like 'meaningfulness' etc.
        for v in d.values():
            if isinstance(v, dict) and 'score_0_100' in v:
                return int(round(float(v['score_0_100'])))
        return np.nan
    except Exception:
        return np.nan

# === Usage ===
# assuming df exists and has columns 'output' and 'meaningfulness' (or your column names)
# df = pd.read_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/meaningfulness_llm_bert_scores_500_Original.csv", encoding="utf-8")
df = pd.read_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/meaningfulness_llm_bert_scores_500.csv", encoding="utf-8")

# create extracted columns
df['output_grade'] = df['output'].apply(extract_grade_cell)
df['output_grade'] = df['output_grade'] * 10
df['meaningfulness_score'] = df['meaningfulness'].apply(extract_meaningfulness_score_cell)

# # optional: create diagnostics to inspect failures
# df['grade_extracted_ok'] = df['output_grade'].notna()
# df['meaningfulness_extracted_ok'] = df['meaningfulness_score'].notna()

# # Save a small sample of failed rows for inspection
# failed = df[~df['grade_extracted_ok'] | ~df['meaningfulness_extracted_ok']]
# if not failed.empty:
#     failed[['output', 'meaningfulness']].to_csv("extraction_failures_sample.csv", index=False, encoding="utf-8")

# final: keep only integer columns if desired
df_clean = df[['output_grade', 'meaningfulness_score']].copy()

# df.to_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/meaningfulness_llm_bert_scores_500_originalv2.csv", index=False, encoding="utf-8")
df.to_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/meaningfulness_llm_bert_scores_500v2.csv", index=False, encoding="utf-8")

# Now you can compute correlations safely (dropna)
pearson = df_clean['output_grade'].corr(df_clean['meaningfulness_score'], method='pearson')
spearman = df_clean['output_grade'].corr(df_clean['meaningfulness_score'], method='spearman')

print("Pearson:", pearson, "Spearman:", spearman)


import matplotlib.pyplot as plt

plt.scatter(df_clean['output_grade'], df_clean['meaningfulness_score'], alpha=0.6)
plt.xlabel("LLM Meaningfulness Score")
plt.ylabel("BERT Meaningfulness Score")
plt.title("LLM vs BERT Meaningfulness Correlation")
# plt.savefig("meaningfulness_bert_llm_original")
plt.savefig("meaningfulness_bert_llm_finetuned")
plt.show()


# Filter rows where output_grade < 30 and meaningfulness_score > 60
filtered_df = df[(df['output_grade'] < 30) & (df['meaningfulness_score'] > 60)]

# Show 10 examples
# filtered_df.to_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/meaningfulness_llm_bert_scores_500_OriginalvControl.csv", index=False, encoding="utf-8")
filtered_df.to_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/meaningfulness_llm_bert_scores_500vControl.csv", index=False, encoding="utf-8")
