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


def filter_truth_by_description(
    csv_path: str,
    json_folder: str,
    output_csv_path: str | None = None,
    truth_col: str = "truth",
    modelid_col: str = "ModelID",
    description_key: str = "description",
    keep_original_truth_format: bool = True,
) -> str:
    """
    Reads csv_path, filters items in truth_col to those present in each matching JSON's description,
    writes a new CSV, prints summary, and returns the output path.

    keep_original_truth_format:
      - True  => writes back a Python-list-like string (e.g. "['a', 'b']")
      - False => writes back a JSON list string (e.g. '["a","b"]')
    """
    df = pd.read_csv(csv_path)

    if truth_col not in df.columns:
        raise ValueError(f"CSV missing required column: {truth_col}")
    if modelid_col not in df.columns:
        raise ValueError(f"CSV missing required column: {modelid_col}")

    removed_counter = Counter()
    kept_counter = Counter()

    total_rows = len(df)
    total_items_before = 0
    total_items_after = 0
    rows_missing_json = 0
    rows_missing_description_key = 0

    new_truth_values = []

    for idx, row in df.iterrows():
        model_id = row[modelid_col]
        truth_items = parse_truth_cell(row[truth_col])
        total_items_before += len(truth_items)

        json_filename = modelid_to_json_filename(model_id)
        json_path = os.path.join(json_folder, json_filename)

        if not os.path.isfile(json_path):
            # If JSON missing, keep truth as-is (or you could choose to empty it)
            rows_missing_json += 1
            filtered = truth_items
            for t in truth_items:
                kept_counter[normalize_text(t)] += 1
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # Bad JSON: keep truth as-is
                    filtered = truth_items
                    for t in truth_items:
                        kept_counter[normalize_text(t)] += 1
                else:
                    desc = data.get(description_key, None)
                    if desc is None:
                        rows_missing_description_key += 1
                        filtered = truth_items
                        for t in truth_items:
                            kept_counter[normalize_text(t)] += 1
                    else:
                        norm_desc = normalize_text(desc)

                        filtered = []
                        for item in truth_items:
                            norm_item = normalize_text(item)
                            if phrase_in_description(norm_item, norm_desc):
                                filtered.append(item)
                                kept_counter[norm_item] += 1
                            else:
                                removed_counter[norm_item] += 1

        total_items_after += len(filtered)

        if keep_original_truth_format:
            # Python-list style string (common in CSV exports)
            new_truth_values.append(str(filtered))
        else:
            # JSON list string
            new_truth_values.append(json.dumps(filtered, ensure_ascii=False))

    df[truth_col] = new_truth_values

    if output_csv_path is None:
        root, ext = os.path.splitext(csv_path)
        output_csv_path = f"{root}_filtered{ext or '.csv'}"

    df.to_csv(output_csv_path, index=False)

    # ---- Summary ----
    removed_total = sum(removed_counter.values())
    kept_total = sum(kept_counter.values())
    denom = total_items_before if total_items_before else 1
    removal_rate = removed_total / denom * 100.0

    print("\n===== FILTER SUMMARY =====")
    print(f"Rows processed: {total_rows}")
    print(f"Total truth items before: {total_items_before}")
    print(f"Total truth items after:  {total_items_after}")
    print(f"Total removed:            {removed_total} ({removal_rate:.2f}%)")
    print(f"Total kept:               {kept_total}")
    print(f"Rows with missing JSON:   {rows_missing_json}")
    print(f"Rows missing '{description_key}' key: {rows_missing_description_key}")

    if removed_counter:
        print("\nTop 20 most frequently removed (normalized) items:")
        for phrase, cnt in removed_counter.most_common(20):
            print(f"  {cnt:>5}  {phrase}")

    # Optional: also show “net removal” vs kept per phrase (if you want)
    # (uncomment if useful)
    # print("\nTop 20 items by removal minus kept (normalized):")
    # net = {k: removed_counter[k] - kept_counter.get(k, 0) for k in removed_counter}
    # for phrase, score in sorted(net.items(), key=lambda x: x[1], reverse=True)[:20]:
    #     print(f"  {score:>5}  {phrase}  (removed={removed_counter[phrase]}, kept={kept_counter.get(phrase,0)})")

    print(f"\nWrote: {output_csv_path}")
    return output_csv_path


# Example usage:
SAMPLE = "T2"
filter_truth_by_description(
    csv_path=f"10-EVALUATION/models_ffs/model_ffs_eval_T{SAMPLE}.csv",
    json_folder="HF-Models-T6",
    output_csv_path=f"10-EVALUATION/models_ffs/model_ffs_eval_T{SAMPLE}_strained.csv"
)
