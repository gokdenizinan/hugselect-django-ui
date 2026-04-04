from __future__ import annotations

import os
import json
from collections import Counter
from typing import Iterable

import pandas as pd

import os
import re
import json
import ast
import unicodedata
from collections import Counter

import pandas as pd


def normalize_text(s: str) -> str:
    """
    Basic normalization:
    - unicode normalize
    - lowercase
    - convert punctuation to spaces
    - collapse whitespace
    """
    if s is None:
        return ""
    s = str(s)

    # Unicode normalize (handles odd quotes/diacritics better)
    s = unicodedata.normalize("NFKC", s)

    s = s.lower()

    # Replace any non-alphanumeric with spaces (keeps letters/numbers)
    # This makes matching more consistent across punctuation differences.
    s = re.sub(r"[^a-z0-9]+", " ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_truth_cell(cell) -> list[str]:
    """
    Parses a CSV cell that contains a list of noun phrases.
    Supports:
    - actual list objects
    - stringified Python list: "['a', 'b']"
    - stringified JSON list: '["a", "b"]'
    - fallback split on common separators: ; | ,  (heuristic)
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []

    if isinstance(cell, list):
        return [str(x) for x in cell]

    s = str(cell).strip()
    if not s:
        return []

    # Try JSON list
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(x) for x in val]
    except Exception:
        pass

    # Try Python literal list
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x) for x in val]
    except Exception:
        pass

    # Heuristic split
    # Prefer semicolon or pipe if present; otherwise comma.
    if ";" in s:
        parts = s.split(";")
    elif "|" in s:
        parts = s.split("|")
    elif "," in s:
        parts = s.split(",")
    else:
        parts = [s]

    return [p.strip() for p in parts if p.strip()]


def modelid_to_json_filename(model_id: str) -> str:
    """
    Replace '/' with '__' and add '.json'
    """
    return str(model_id).replace("/", "__") + ".json"


def phrase_in_description(norm_phrase: str, norm_description: str) -> bool:
    """
    Check if a normalized phrase exists in a normalized description.

    Uses word-boundary-aware regex to reduce false positives.
    Example: phrase 'cat' should not match 'concatenate'.

    For multi-word phrases, boundaries still work well because words/spaces are preserved.
    """
    if not norm_phrase:
        return False

    # Word boundary-ish: phrase must not be preceded/followed by a-z0-9
    # (since our normalization reduces everything to a-z0-9 and spaces)
    pattern = r"(?<![a-z0-9])" + re.escape(norm_phrase) + r"(?![a-z0-9])"
    return re.search(pattern, norm_description) is not None


def filter_columns_by_description(
    csv_path: str,
    json_folder: str,
    output_csv_path: str | None = None,
    truth_cols: list[str] | None = None,
    modelid_col: str = "ModelID",
    description_key: str = "description",
    keep_original_truth_format: bool = True,
) -> str:
    """
    Reads csv_path, filters items in each column in truth_cols to those present in each matching JSON's description,
    writes a new CSV, prints summary, and returns the output path.

    keep_original_truth_format:
      - True  => writes back a Python-list-like string (e.g. "['a', 'b']")
      - False => writes back a JSON list string (e.g. '["a","b"]')
    """
    if truth_cols is None:
        truth_cols = ["Gemini_Features", "ChatGPT_Features", "Instruct_Features"]

    df = pd.read_csv(csv_path)

    # Validate columns
    if modelid_col not in df.columns:
        raise ValueError(f"CSV missing required column: {modelid_col}")
    missing_truth_cols = [c for c in truth_cols if c not in df.columns]
    if missing_truth_cols:
        raise ValueError(f"CSV missing required columns: {missing_truth_cols}")

    # Per-column counters
    removed_counter = {c: Counter() for c in truth_cols}
    kept_counter = {c: Counter() for c in truth_cols}

    total_rows = len(df)
    total_items_before = {c: 0 for c in truth_cols}
    total_items_after = {c: 0 for c in truth_cols}

    rows_missing_json = 0
    rows_missing_description_key = 0
    rows_bad_json = 0

    # We'll build new column values as lists, then assign at the end
    new_values = {c: [] for c in truth_cols}

    for _, row in df.iterrows():
        model_id = row[modelid_col]
        json_filename = modelid_to_json_filename(model_id)
        json_path = os.path.join(json_folder, json_filename)

        # Load description once per row
        desc = None
        if not os.path.isfile(json_path):
            rows_missing_json += 1
        else:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                desc = data.get(description_key, None)
                if desc is None:
                    rows_missing_description_key += 1
            except json.JSONDecodeError:
                rows_bad_json += 1

        norm_desc = normalize_text(desc) if isinstance(desc, str) else None

        # Filter each requested column
        for col in truth_cols:
            items = parse_truth_cell(row[col])
            total_items_before[col] += len(items)

            if norm_desc is None:
                # Keep as-is if JSON missing/bad/description missing
                filtered = items
                for t in items:
                    kept_counter[col][normalize_text(t)] += 1
            else:
                filtered = []
                for item in items:
                    norm_item = normalize_text(item)
                    if phrase_in_description(norm_item, norm_desc):
                        filtered.append(item)
                        kept_counter[col][norm_item] += 1
                    else:
                        removed_counter[col][norm_item] += 1

            total_items_after[col] += len(filtered)

            if keep_original_truth_format:
                new_values[col].append(str(filtered))
            else:
                new_values[col].append(json.dumps(filtered, ensure_ascii=False))

    # Assign updated columns
    for col in truth_cols:
        df[col] = new_values[col]

    if output_csv_path is None:
        root, ext = os.path.splitext(csv_path)
        output_csv_path = f"{root}_filtered{ext or '.csv'}"

    df.to_csv(output_csv_path, index=False)

    # ---- Summary ----
    print("\n===== FILTER SUMMARY (MULTI-COLUMN) =====")
    print(f"Rows processed: {total_rows}")
    print(f"Rows with missing JSON:              {rows_missing_json}")
    print(f"Rows with bad JSON:                  {rows_bad_json}")
    print(f"Rows missing '{description_key}' key: {rows_missing_description_key}")

    for col in truth_cols:
        removed_total = sum(removed_counter[col].values())
        kept_total = sum(kept_counter[col].values())
        denom = total_items_before[col] if total_items_before[col] else 1
        removal_rate = removed_total / denom * 100.0

        print(f"\n--- Column: {col} ---")
        print(f"Total items before: {total_items_before[col]}")
        print(f"Total items after:  {total_items_after[col]}")
        print(f"Total removed:      {removed_total} ({removal_rate:.2f}%)")
        print(f"Total kept:         {kept_total}")

        if removed_counter[col]:
            print("Top 5 most frequently removed (normalized):")
            for phrase, cnt in removed_counter[col].most_common(5):
                print(f"  {cnt:>5}  {phrase}")

    print(f"\nWrote: {output_csv_path}")
    return output_csv_path


# UNTED COLUMN

from collections import Counter
import json
import os
import pandas as pd

import re

from collections import Counter
import json
import os
import pandas as pd
import re


def items_match(a: str, b: str) -> bool:
    def normalize(x: str) -> str:
        x = x.lower().strip()
        x = re.sub(r"[^\w\s]", "", x)
        x = re.sub(r"\s+", " ", x)
        return x

    return normalize(a) == normalize(b)


def create_united_features_column(
    csv_path: str,
    output_csv_path: str | None = None,
    truth_cols: list[str] | None = None,
    new_col: str = "United_Features",
    keep_original_truth_format: bool = True,
) -> str:

    if truth_cols is None:
        truth_cols = ["Gemini_Features", "ChatGPT_Features", "Instruct_Features"]

    df = pd.read_csv(csv_path)

    united_values = []

    total_rows = len(df)
    total_items_before = 0
    total_items_after = 0

    for _, row in df.iterrows():

        lists = [parse_truth_cell(row[col]) for col in truth_cols]

        kept_items = []

        # Count original total (sum of all 3 columns)
        total_items_before += sum(len(lst) for lst in lists)

        all_items = []
        for lst in lists:
            all_items.extend(lst)

        for item in all_items:

            count = 0
            for col_list in lists:
                if any(items_match(item, other) for other in col_list):
                    count += 1

            if count >= 2:
                if not any(items_match(item, existing) for existing in kept_items):
                    kept_items.append(item)

        total_items_after += len(kept_items)

        if keep_original_truth_format:
            united_values.append(str(kept_items))
        else:
            united_values.append(json.dumps(kept_items, ensure_ascii=False))

    df.insert(loc=9, column=new_col, value=united_values)

    if output_csv_path is None:
        root, ext = os.path.splitext(csv_path)
        output_csv_path = f"{root}_with_united{ext or '.csv'}"

    df[new_col] = df[new_col].apply(parse_truth_cell)
    if new_col in df.columns:
        df.insert(loc=10, column=f"{new_col[:4]}_count", value=df[new_col].apply(len))


    df.to_csv(output_csv_path, index=False)

    # ===== REPORT =====
    removed = total_items_before - total_items_after
    removal_rate = (removed / total_items_before * 100) if total_items_before else 0

    print("\n===== UNITED FEATURES AGREEMENT REPORT =====")
    print(f"Rows processed: {total_rows}")
    print(f"Total items across 3 columns: {total_items_before}")
    print(f"Items kept (>=2/3 agreement): {total_items_after}")
    print(f"Items removed (no agreement): {removed}")
    print(f"Removal rate: {removal_rate:.2f}%")
    print(f"Wrote: {output_csv_path}\n")

    return output_csv_path


INPUT_PATH= "10-EVALUATION/enriched_ffs/enriched_ffs_eval_T5.csv"
OUT_PATH = "10-EVALUATION/enriched_ffs/enriched_ffs_eval_T5_strained.csv"
filter_columns_by_description(
    csv_path=INPUT_PATH,
    json_folder="HF-Models-T6",
    truth_cols=["Gemini_Features", "ChatGPT_Features", "Instruct_Features"],
    output_csv_path=OUT_PATH,
    keep_original_truth_format=True,
)

INPUT_PATH_UNITED = "10-EVALUATION/enriched_ffs/enriched_ffs_eval_T5_strained.csv"
OUT_PATH_UNITED = "10-EVALUATION/enriched_ffs/enriched_ffs_eval_T5_united.csv"
create_united_features_column(
    csv_path=INPUT_PATH_UNITED,
    output_csv_path=OUT_PATH_UNITED
)
