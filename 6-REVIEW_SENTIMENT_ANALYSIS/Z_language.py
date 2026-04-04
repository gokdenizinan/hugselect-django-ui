from __future__ import annotations
import json
from pathlib import Path
from collections import Counter


# # Folder containing the dictionary JSON files
# input_folder = Path("HF-Models-T6")

# output_file = Path("language_counts.json")

# language_counter = Counter()

# for json_file in input_folder.glob("*.json"):
#     with open(json_file, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     language = data.get("Metadata", {}).get("language")

#     if language is None:
#         continue

#     # If language is a list, count each element
#     if isinstance(language, list):
#         for lang in language:
#             language_counter[lang] += 1

#     # If language is a single value
#     else:
#         language_counter[language] += 1

# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(language_counter, f, indent=2, ensure_ascii=False)

# print("Saved language counts to", output_file)

import csv
def load_iso6393(path="8-CRITERIA_SELECTION/iso-639-3.tab"):
    mapping = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mapping[row["Id"]] = row["Ref_Name"]
    return mapping

# iso_map = load_iso6393()

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List


@dataclass(frozen=True)
class LangHit:
    iso639_3: str
    name: str
    source: str  # "Id", "Part1", "Part2T", "Part2B", "special", "name-match"


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    # normalize separators
    s = s.replace("_", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def load_iso6393_table(iso_tab_path: str | Path) -> Dict[str, Dict[str, str]]:
    """
    Loads iso-639-3.tab (tab-separated) from SIL.
    Returns dict keyed by Id with the row dict as value.
    """
    iso_tab_path = Path(iso_tab_path)
    rows_by_id: Dict[str, Dict[str, str]] = {}
    with iso_tab_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("Id"):
                rows_by_id[row["Id"].strip()] = row
    return rows_by_id


def build_indexes(rows_by_id: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, LangHit], Dict[str, LangHit]]:
    """
    Builds:
      - code_index: maps known codes (Id/Part1/Part2T/Part2B) -> LangHit
      - name_index: maps normalized language names -> LangHit
    """
    code_index: Dict[str, LangHit] = {}
    name_index: Dict[str, LangHit] = {}

    for iso3, row in rows_by_id.items():
        name = row.get("Ref_Name", "").strip()
        if not name:
            continue

        # Id (ISO 639-3)
        code_index[_norm(iso3)] = LangHit(iso3, name, "Id")

        # Part1 (ISO 639-1)
        p1 = row.get("Part1", "").strip()
        if p1:
            code_index[_norm(p1)] = LangHit(iso3, name, "Part1")

        # Part2T / Part2B (ISO 639-2)
        p2t = row.get("Part2T", "").strip()
        if p2t:
            code_index[_norm(p2t)] = LangHit(iso3, name, "Part2T")

        p2b = row.get("Part2B", "").strip()
        if p2b:
            code_index[_norm(p2b)] = LangHit(iso3, name, "Part2B")

        # Name index (exact normalized names)
        name_index[_norm(name)] = LangHit(iso3, name, "name-match")

    return code_index, name_index


# Common HF / informal / BCP-47-ish specials.
SPECIAL_TAGS: Dict[str, Tuple[str, str]] = {
    "zh-": ("zho", "Chinese"),
    "zhs": ("zho", "Chinese (Simplified)"),
    "zht": ("zho", "Chinese (Traditional)"),
    # Common region/script tags (treat as base language)
    "zh-cn": ("zho", "Chinese"),
    "zh-hans": ("zho", "Chinese (Simplified)"),
    "zh-hant": ("zho", "Chinese (Traditional)"),
}

# Extra name/alias synonyms for “snap to base language” from free text
# (add more if your data has more messy tags).
NAME_SYNONYMS: Dict[str, str] = {
    "farsi": "Persian",
    "dari": "Persian",
    "pashto": "Pashto",
    "mandarin": "Chinese",
    "cantonese": "Chinese",
    "simplified chinese": "Chinese (Simplified)",
    "traditional chinese": "Chinese (Traditional)",
    "brazilian portuguese": "Portuguese",
    "mexican spanish": "Spanish",
    "cancun french": "French",  # example you gave
}


def resolve_language(tag: str, code_index: Dict[str, LangHit], name_index: Dict[str, LangHit]) -> Optional[LangHit]:
    """
    Resolve a Hugging Face-ish language tag / alias / free text to a language name using ISO table.
    Returns LangHit or None if unknown.
    """
    raw = tag or ""
    t = _norm(raw)

    if not t:
        return None

    # 1) Direct special tag handling
    if t in SPECIAL_TAGS:
        iso3, display = SPECIAL_TAGS[t]
        return LangHit(iso3, display, "special")

    # 2) If it looks like a BCP-47 tag (en-us, pt-br, zh-hant...), take primary subtag
    # Also handles underscores because we normalized to '-'
    primary = t.split("-")[0]

    # 3) Exact code match (Id / Part1 / Part2T / Part2B)
    if t in code_index:
        return code_index[t]
    if primary in code_index:
        return code_index[primary]

    # 4) Exact name match (rare but useful if someone wrote "French")
    if t in name_index:
        return name_index[t]

    # 5) Synonym snapping (e.g., "cancun french" -> "French")
    if t in NAME_SYNONYMS:
        snapped = _norm(NAME_SYNONYMS[t])
        if snapped in name_index:
            return name_index[snapped]
        # If synonym is a code like "fr"
        if snapped in code_index:
            return code_index[snapped]

    # 6) Free-text heuristic: find a language name contained in the text.
    # Example: "cancun french" contains "french"
    # We prefer the LONGEST match to avoid tiny collisions.
    text = f" {t} "
    best: Optional[LangHit] = None
    best_len = 0

    # First try synonym keys contained in text
    for k, v in NAME_SYNONYMS.items():
        kn = _norm(k)
        if f" {kn} " in text or kn in t:
            vn = _norm(v)
            hit = name_index.get(vn)
            if hit and len(kn) > best_len:
                best = hit
                best_len = len(kn)

    # Then try ISO Ref_Name contained in text (can be heavy but OK for small/medium use)
    for nm, hit in name_index.items():
        if len(nm) < 4:
            continue
        if f" {nm} " in text:
            if len(nm) > best_len:
                best = hit
                best_len = len(nm)

    return best


import json
from pathlib import Path

def new_name(value,code_index, name_index ):
    # example transformation
    hit = resolve_language(value, code_index, name_index)
    if hit:
        hit_val = hit.name
        hit_low = hit_val.lower()
        return hit_low
    else:
        return "UNKNOWN"





if __name__ == "__main__":
    # 1) Download iso-639-3.tab from https://iso639-3.sil.org/code_tables/download_tables
    # 2) Put it next to this script or pass the path below.
    rows = load_iso6393_table("8-CRITERIA_SELECTION/iso-639-3.tab")
    code_index, name_index = build_indexes(rows)

    tests = ["fr", "fra", "fre", "en-us", "pt_BR", "zhs", "zh-hant", "cancun french", "French"]
    for x in tests:
        hit = resolve_language(x, code_index, name_index)
        print("")
        print(f"{x!r} -> {hit.name if hit else 'UNKNOWN'}  (iso639-3={hit.iso639_3 if hit else 'n/a'}, via={hit.source if hit else 'n/a'})")
        print(new_name(x,code_index,name_index))
    
    
    hit = resolve_language("simplified chinese", code_index, name_index)
    print(hit.name.lower())

    input_folder = Path("path/to/your/json_folder")

    for json_file in input_folder.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        language = metadata.get("language")

        if language is None:
            continue

        # Handle list of languages
        if isinstance(language, list):
            metadata["language"] = [new_name(lang) for lang in language]

        # Handle single language
        else:
            metadata["language"] = new_name(language)

        # Save back to the same file
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print("All language values replaced.")