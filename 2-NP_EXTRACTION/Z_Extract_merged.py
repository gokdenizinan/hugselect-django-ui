"""
noun_phrase_pipeline.py

Full, modified code:
- Object-oriented noun phrase extractor
- Per-filter on/off toggles
- Configurable useful concepts set
- Single-text extraction (row-per-NP)
- Folder processing:
    1) row-per-NP DataFrame with NP-relative entities + concept mappings
    2) optional row-per-file (wide) DataFrame with list columns aligned by NP index
- NP aggregation function (df -> aggregated df)

REQUIREMENTS:
- pip install spacy pandas emoji tqdm
- python -m spacy download en_core_web_sm
- Ab_generic_terms.py must define: GENERIC_TERMS, EXCLUDE, WEAK_HEADS
"""

from __future__ import annotations

import os
from tqdm import tqdm
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from collections import Counter

import pandas as pd
import spacy
import emoji
from spacy.lang.en.stop_words import STOP_WORDS

from Ab_generic_terms import GENERIC_TERMS, EXCLUDE, WEAK_HEADS


# -------------------------
# Concept mapping
# -------------------------
ENTITY_TO_CONCEPT: Dict[str, str] = {
    "ORG": "Tool",
    "PERSON": "Contributor",
    "NORP": "Language Group",
    "GPE": "Locale",
    "LOC": "Locale",
    "LANGUAGE": "Language",
    "PRODUCT": "Model Component",
    "FAC": "Infrastructure",
    "WORK_OF_ART": "Dataset",
    "EVENT": "Benchmark",
    "LAW": "License",
    "DATE": "Training Info",
    "TIME": "Training Info",
    "ORDINAL": "Metric",
    "CARDINAL": "Metric",
    "MONEY": "Training Info",
    "PERCENT": "Metric",
    "QUANTITY": "Training Info",
}

DEFAULT_USEFUL_CONCEPTS: Set[str] = {
    "Tool",
    "Dataset",
    "Model Component",
    "Infrastructure",
    "Benchmark",
    "Language Group",
    "License",
}

LATIN_REGEX = re.compile(r"^[\u0041-\u024F]+$", re.UNICODE)


def map_entity_to_concept(label: str) -> str:
    return ENTITY_TO_CONCEPT.get(label, "Other")


def _mode(values: Iterable[Any]) -> Any:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    c = Counter(vals)
    return c.most_common(1)[0][0]


# -------------------------
# Config
# -------------------------
@dataclass
class NounPhraseExtractorConfig:
    # spaCy
    spacy_model: str = "en_core_web_sm"

    # Per-filter toggles (NP filtering)
    enable_validity_filter: bool = True
    enable_structure_filter: bool = True
    enable_concept_filter: bool = True
    enable_english_filter: bool = True
    enable_multiword_filter: bool = False  # if True => enforce multiword-only

    # Concept criteria (used only if enable_concept_filter=True)
    useful_concepts: Set[str] = field(default_factory=lambda: set(DEFAULT_USEFUL_CONCEPTS))

    # Passing rule:
    # - None: must pass ALL enabled filters (matches your original "good_np < 4" idea, but dynamic)
    # - int: must pass at least N enabled filters
    require_pass_count: Optional[int] = None

    # Entity-in-NP matching mode
    # - "contain": entity span must be fully inside NP span
    # - "overlap": any overlap counts
    entity_match_mode: str = "contain"  # "contain" or "overlap"


# -------------------------
# Extractor
# -------------------------
class NounPhraseExtractor:
    """
    Core methods:
    - extract_from_text(text) -> list[dict] (row-per-NP)
    - process_folder_np_relative_entities_df(...) -> pd.DataFrame
      * output="long": one row per NP per file (recommended)
      * output="wide": one row per file with list-columns aligned by NP index
    - aggregate_noun_phrases_df(df, min_global_freq=10) -> pd.DataFrame
    """

    def __init__(self, config: Optional[NounPhraseExtractorConfig] = None):
        self.config = config or NounPhraseExtractorConfig()
        self.nlp = spacy.load(self.config.spacy_model)
        self.reset_stats()

    def reset_stats(self) -> None:
        self.np_seen: set[str] = set()

        self.filter_counts = {
            "Total": 0,
            "Validity": 0,
            "Structure": 0,
            "Concept": 0,
            "English": 0,
            "Multiword": 0,
            "Passed": 0,
        }
        self.filter_counts_set = {k: 0 for k in self.filter_counts.keys()}

        self.concept_counts = {
            "NONE": 0,
            "PERSON": 0,
            "GPE": 0,
            "LOC": 0,
            "DATE": 0,
            "TIME": 0,
            "ORDINAL": 0,
            "CARDINAL": 0,
            "MONEY": 0,
            "PERCENT": 0,
            "LANGUAGE": 0,
            "FAC": 0,
            "LAW": 0,
            "QUANTITY": 0,
            "ORG": 0,
            "NORP": 0,
            "PRODUCT": 0,
            "WORK_OF_ART": 0,
            "EVENT": 0,
        }

    # -------------------------
    # Filters (ported)
    # -------------------------

    def is_valid_np(self, np_text: str) -> bool:
        np_text = np_text.strip().lower()
        URL_REGEX = re.compile(r"https?://|www\.", re.IGNORECASE)

        # NEW: reject URLs
        if URL_REGEX.search(np_text):
            return False

        if re.fullmatch(r"\W+", np_text):
            return False
        if np_text in STOP_WORDS:
            return False
        if np_text in GENERIC_TERMS:
            return False
        if len(np_text.split()) == 1 and len(np_text) < 3:
            return False
        if np_text.isdigit():
            return False
        if any(np_text.startswith(det + " ") for det in {"the", "this", "that", "these", "those", "a", "an"}):
            return False
        if any(np_text.startswith(p.lower()) for p in EXCLUDE):
            return False
        return True

    def is_structurally_useful_np(self, chunk) -> bool:
        if all(tok.pos_ in {"ADJ", "ADV"} for tok in chunk):
            return False
        for tok in chunk:
            if tok.dep_ == "amod" and tok.head.pos_ == "ADJ":
                return False
        if len(chunk) <= 2 and chunk.root.lemma_.lower() in WEAK_HEADS:
            return False
        return any(tok.pos_ in {"NOUN", "PROPN", "NUM"} for tok in chunk)

    def is_multiword(self, np_text: str) -> bool:
        if self.config.enable_multiword_filter and len(np_text.split()) <= 1:
            return False
        return True

    def is_english_like(self, np_text: str) -> bool:
        if any(ch in emoji.EMOJI_DATA for ch in np_text):
            return False

        for tok in np_text.split():
            tok_clean = re.sub(r"\W+", "", tok)
            if not tok_clean:
                continue
            if tok_clean.isalpha() and not LATIN_REGEX.match(tok_clean):
                return False
        return True

    def _required_passes(self) -> int:
        enabled = 0
        enabled += int(self.config.enable_validity_filter)
        enabled += int(self.config.enable_structure_filter)
        enabled += int(self.config.enable_concept_filter)
        enabled += int(self.config.enable_english_filter)
        enabled += int(self.config.enable_multiword_filter)

        if self.config.require_pass_count is None:
            return enabled  # all enabled must pass
        return min(int(self.config.require_pass_count), enabled)

    # -------------------------
    # NP modifiers helper
    # -------------------------
    def _extract_modifiers_from_chunk(self, chunk) -> List[str]:
        """
        Modifiers for the NP (excluding head token).
        Includes common modifier deps: amod, compound, nummod, poss, det.
        """
        head = chunk.root
        mods: List[str] = []
        for tok in chunk:
            if tok.i == head.i:
                continue
            if tok.dep_ in {"amod", "compound", "nummod", "poss", "det"}:
                mods.append(tok.text)

        # de-dup while preserving order
        seen = set()
        out: List[str] = []
        for m in mods:
            if m not in seen:
                out.append(m)
                seen.add(m)
        return out

    # -------------------------
    # Entity-in-NP helper
    # -------------------------
    def _entities_in_np_span(self, doc, start: int, end: int):
        """
        Return list of entities considered "inside" NP span.

        If entity_match_mode == "contain":
          ent.start >= start AND ent.end <= end

        If entity_match_mode == "overlap":
          ent.start < end AND ent.end > start
        """
        mode = (self.config.entity_match_mode or "contain").lower()
        ents = []
        for ent in doc.ents:
            if mode == "overlap":
                if ent.start < end and ent.end > start:
                    ents.append(ent)
            else:
                if ent.start >= start and ent.end <= end:
                    ents.append(ent)
        return ents

    # -------------------------
    # Single text NP extraction (filtered, row-per-NP)
    # -------------------------
    def extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text or "")
        ent_span_to_label = {(ent.start, ent.end): ent.label_ for ent in doc.ents}
        key_entity_labels = set(ENTITY_TO_CONCEPT.keys())
        required = self._required_passes()

        entries: List[Dict[str, Any]] = []

        for chunk in doc.noun_chunks:
            self.filter_counts["Total"] += 1

            phrase = chunk.text.strip()
            if not phrase:
                continue

            is_new_phrase = phrase not in self.np_seen
            if is_new_phrase:
                self.filter_counts_set["Total"] += 1

            passes = 0

            # Multiword
            if self.config.enable_multiword_filter:
                if self.is_multiword(phrase):
                    self.filter_counts["Multiword"] += 1
                    passes += 1
                    if is_new_phrase:
                        self.filter_counts_set["Multiword"] += 1
                else:
                    self.np_seen.add(phrase)
                    continue

            # Validity
            if self.config.enable_validity_filter and self.is_valid_np(phrase):
                self.filter_counts["Validity"] += 1
                passes += 1
                if is_new_phrase:
                    self.filter_counts_set["Validity"] += 1

            # Structure
            if self.config.enable_structure_filter and self.is_structurally_useful_np(chunk):
                self.filter_counts["Structure"] += 1
                passes += 1
                if is_new_phrase:
                    self.filter_counts_set["Structure"] += 1

            # Concept (based on EXACT NP == entity span, matching your original approach)
            entity_label = ent_span_to_label.get((chunk.start, chunk.end), "O")
            if is_new_phrase:
                if entity_label in key_entity_labels:
                    self.concept_counts[entity_label] += 1
                else:
                    self.concept_counts["NONE"] += 1

            concept = map_entity_to_concept(entity_label)
            if self.config.enable_concept_filter and (concept in self.config.useful_concepts):
                self.filter_counts["Concept"] += 1
                passes += 1
                if is_new_phrase:
                    self.filter_counts_set["Concept"] += 1

            # English-like
            if self.config.enable_english_filter and self.is_english_like(phrase):
                self.filter_counts["English"] += 1
                passes += 1
                if is_new_phrase:
                    self.filter_counts_set["English"] += 1

            self.np_seen.add(phrase)

            if passes < required:
                continue

            self.filter_counts["Passed"] += 1
            if is_new_phrase:
                self.filter_counts_set["Passed"] += 1

            entries.append(
                {
                    "noun_phrase": phrase,
                    "head_noun": chunk.root.text,
                    "root": chunk.root.pos_,
                    "position_in_text": round((chunk.start / len(doc)) if len(doc) else 0.0, 3),
                    "entity_label": entity_label,
                    "concept": concept,
                    "sentence": chunk.sent.text.strip(),
                    "passes": passes,
                    "required": required,
                }
            )

        return entries

    # -------------------------
    # Folder processing: NP-relative entities -> DF
    # -------------------------
    def process_folder_np_relative_entities_df(
        self,
        directory: str,
        description_key: str = "description",
        file_suffix: str = ".json",
        save_path: Optional[str] = None,
        include_only_passing_nps: bool = True,
        output: str = "long",  # "long" or "wide"
    ) -> pd.DataFrame:
        """
        Entity extraction RELATIVE TO NPs.

        LONG output (recommended): one row per NP per file:
          modelId, NP, Head, Modifiers, Entities, Entity Label, Entity Concept,
          #OF Entities (NP), #OF Noun Phrases (file), #OF Entities (file)

        WIDE output: one row per file, with list columns aligned by NP index:
          modelId, NP(list), Head(list), Modifiers(list-of-lists),
          Entities(list-of-lists), Entity Label(list-of-lists), Entity Concept(list-of-lists),
          #OF Noun Phrases (file), #OF Entities (file)
        """
        long_rows: List[Dict[str, Any]] = []
        wide_rows: List[Dict[str, Any]] = []

        files = sorted(
            f for f in os.listdir(directory)
            if f.endswith(file_suffix)
        )

        sample_fraction = 0.01  # 1%
        sample_size = max(1, int(len(files) * sample_fraction))
        files = files[:sample_size]

        for filename in files:
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            model_id = data.get("modelId", filename.replace(file_suffix, ""))
            text = data.get(description_key, "") or ""
            doc = self.nlp(text)

            # Determine which NPs to keep based on your filter settings
            if include_only_passing_nps:
                kept_entries = self.extract_from_text(text)
                kept_set = set(e["noun_phrase"] for e in kept_entries)
            else:
                kept_set = None  # keep all noun_chunks

            np_texts: List[str] = []
            heads: List[str] = []
            modifiers_all: List[List[str]] = []
            ents_all: List[List[str]] = []
            ent_labels_all: List[List[str]] = []
            ent_concepts_all: List[List[str]] = []

            file_entity_seen: set[Tuple[int, int, str]] = set()

            for chunk in doc.noun_chunks:
                np_text = chunk.text.strip()
                if not np_text:
                    continue
                if kept_set is not None and np_text not in kept_set:
                    continue

                head = chunk.root.text
                modifiers = self._extract_modifiers_from_chunk(chunk)

                ents_in_np = self._entities_in_np_span(doc, chunk.start, chunk.end)
                ent_texts = [ent.text for ent in ents_in_np]
                ent_labels = [ent.label_ for ent in ents_in_np]
                ent_concepts = [map_entity_to_concept(lbl) for lbl in ent_labels]

                for ent in ents_in_np:
                    file_entity_seen.add((ent.start, ent.end, ent.label_))

                np_texts.append(np_text)
                heads.append(head)
                modifiers_all.append(modifiers)
                ents_all.append(ent_texts)
                ent_labels_all.append(ent_labels)
                ent_concepts_all.append(ent_concepts)

            num_nps_file = len(np_texts)
            num_ents_file = len(file_entity_seen)

            if output.lower() == "wide":
                wide_rows.append(
                    {
                        "modelId": model_id,
                        "NP": np_texts,
                        "Head": heads,
                        "Modifiers": modifiers_all,
                        "Entities": ents_all,
                        "Entity Label": ent_labels_all,
                        "Entity Concept": ent_concepts_all,
                        "#OF Noun Phrases (file)": num_nps_file,
                        "#OF Entities (file)": num_ents_file,
                    }
                )
            else:
                for i in range(num_nps_file):
                    long_rows.append(
                        {
                            "modelId": model_id,
                            "NP": np_texts[i],
                            "Head": heads[i],
                            "Modifiers": modifiers_all[i],
                            "Entities": ents_all[i],
                            "Entity Label": ent_labels_all[i],
                            "Entity Concept": ent_concepts_all[i],
                            "#OF Entities (NP)": len(ents_all[i]),
                            "#OF Noun Phrases (file)": num_nps_file,
                            "#OF Entities (file)": num_ents_file,
                        }
                    )

        df = pd.DataFrame(wide_rows if output.lower() == "wide" else long_rows)

        if save_path:
            ext = os.path.splitext(save_path)[1].lower()
            if ext == ".csv":
                df.to_csv(save_path, index=True)
            elif ext == ".parquet":
                df.to_parquet(save_path, index=False)
            else:
                df.to_pickle(save_path)

        return df

    # -------------------------
    # Aggregation (NP rows DF -> aggregated DF)
    # -------------------------
    def aggregate_noun_phrases_df(self, df: pd.DataFrame, min_global_freq: int = 10) -> pd.DataFrame:
        """
        Aggregates row-per-NP df (like extract_from_text/process_folder_to_df output).
        If you used process_folder_np_relative_entities_df(output="long"), that DF uses column "NP"
        instead of "noun_phrase". This function supports BOTH.

        Output columns:
          noun_phrase, global_frequency, author(list?) etc...
        Here we aggregate only fields that exist in the DF.
        """
        if df.empty:
            return df.copy()

        # Normalize noun phrase column name
        if "noun_phrase" in df.columns:
            phrase_col = "noun_phrase"
        elif "NP" in df.columns:
            phrase_col = "NP"
        else:
            raise ValueError("DF must contain either 'noun_phrase' or 'NP' column.")

        grouped = df.groupby(phrase_col, dropna=False)

        # Build aggregation dict dynamically based on available columns
        agg_spec: Dict[str, Any] = {
            "global_frequency": (phrase_col, "size"),
        }

        if "modelId" in df.columns:
            agg_spec["modelId"] = ("modelId", lambda s: sorted(set(s.dropna().tolist())))
        if "model_id" in df.columns:
            agg_spec["model_id"] = ("model_id", lambda s: sorted(set(s.dropna().tolist())))
        if "author" in df.columns:
            agg_spec["author"] = ("author", lambda s: sorted(set(s.dropna().tolist())))
        if "sentence" in df.columns:
            agg_spec["sentence"] = ("sentence", lambda s: sorted(set(s.dropna().tolist())))
        if "model_task" in df.columns:
            agg_spec["model_task"] = ("model_task", lambda s: sorted(set(s.dropna().tolist())))
        if "Head" in df.columns:
            agg_spec["Head"] = ("Head", lambda s: _mode(s.dropna().tolist()))
        if "head_noun" in df.columns:
            agg_spec["head_noun"] = ("head_noun", lambda s: _mode(s.dropna().tolist()))
        if "root" in df.columns:
            agg_spec["root"] = ("root", lambda s: _mode(s.dropna().tolist()))
        if "Entity Label" in df.columns:
            # list column -> flatten
            agg_spec["Entity Label"] = ("Entity Label", lambda s: _mode([x for lst in s.dropna().tolist() for x in lst]))
        if "Entity Concept" in df.columns:
            agg_spec["Entity Concept"] = (
                "Entity Concept",
                lambda s: _mode([x for lst in s.dropna().tolist() for x in lst]),
            )

        # position (mean if numeric)
        if "position_in_text" in df.columns:
            agg_spec["position_in_text"] = ("position_in_text", "mean")

        agg = grouped.agg(**agg_spec).reset_index()

        # Rename group key to a standard name
        if phrase_col != "noun_phrase":
            agg = agg.rename(columns={phrase_col: "noun_phrase"})

        if "position_in_text" in agg.columns:
            agg["position_in_text"] = agg["position_in_text"].round(3)

        agg = agg[agg["global_frequency"] >= int(min_global_freq)].copy()
        agg = agg.sort_values(["global_frequency", "noun_phrase"], ascending=[False, True]).reset_index(drop=True)
        return agg

from collections import defaultdict
import os
import json


def build_global_np_dictionary(
    extractor,
    directory: str,
    description_key: str = "description",
    file_suffix: str = ".json",
    min_keep: int = 5,
    sim_threshold: float = 0.80,
    merge_variants: bool = True,
):
    """
    Returns a dictionary of noun phrases aggregated across all files.

    Keeps:
      - noun phrases with total count >= min_keep
      - (optionally) "variant" noun phrases with count < min_keep that are
        similar (>= sim_threshold) to a frequent noun phrase, by MERGING the
        variant into the frequent phrase.

    Output format:
    {
        "NP_1": {
            "noun_phrase": "...",
            "count": int,
            "model_id": [list of model ids],
            "sentence": [list of sentences],
            "merged_from": [list of merged variant strings]   # only present if any merged
        },
        ...
    }
    """
    import os
    import json
    import re
    from collections import defaultdict
    from difflib import SequenceMatcher

    def _norm_np(s: str) -> str:
        s = (s or "").strip().lower()
        # keep letters/numbers, turn separators into spaces
        s = re.sub(r"[^a-z0-9]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _sim(a: str, b: str) -> float:
        return SequenceMatcher(None, _norm_np(a), _norm_np(b)).ratio()

    # Aggregate noun phrases globally
    agg = defaultdict(lambda: {"count": 0, "model_id": set(), "sentence": set(), "merged_from": set()})

    files = sorted(f for f in os.listdir(directory) if f.endswith(file_suffix))

    for filename in files:
        filepath = os.path.join(directory, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # Skip unreadable files
            continue

        # Model id from filename without suffix
        model_id = filename[: -len(file_suffix)] if filename.endswith(file_suffix) else filename

        text = data.get(description_key, "") or ""

        # Extract noun phrases from this file
        np_entries = extractor.extract_from_text(text)

        for entry in np_entries:
            np_text = entry["noun_phrase"]
            sentence = entry["sentence"]

            agg[np_text]["count"] += 1
            agg[np_text]["model_id"].add(model_id)
            agg[np_text]["sentence"].add(sentence)

    # Identify frequent phrases
    frequent = {np_text for np_text, v in agg.items() if v["count"] >= min_keep}

    if merge_variants and frequent:
        # Bucket frequent phrases by first normalized token for speed
        buckets = {}
        for f in frequent:
            norm = _norm_np(f)
            key = norm.split(" ", 1)[0] if norm else ""
            buckets.setdefault(key, []).append(f)

        # Merge rare phrases into best matching frequent phrase
        # Work on a stable list so we can delete keys safely
        for np_text in list(agg.keys()):
            v = agg[np_text]
            if v["count"] >= min_keep:
                continue

            norm = _norm_np(np_text)
            key = norm.split(" ", 1)[0] if norm else ""
            candidates = buckets.get(key) or list(frequent)

            best_score = 0.0
            best_match = None
            for f in candidates:
                s = _sim(np_text, f)
                if s > best_score:
                    best_score = s
                    best_match = f

            if best_match is not None and best_score >= sim_threshold:
                # Merge into the frequent canonical phrase
                agg[best_match]["count"] += v["count"]
                agg[best_match]["model_id"].update(v["model_id"])
                agg[best_match]["sentence"].update(v["sentence"])
                agg[best_match]["merged_from"].add(np_text)

                # If this phrase already had merges, carry them over too
                if v.get("merged_from"):
                    agg[best_match]["merged_from"].update(v["merged_from"])

                # Remove the variant key
                del agg[np_text]

        # Recompute frequent after merges (counts may have changed)
        frequent = {np_text for np_text, v in agg.items() if v["count"] >= min_keep}

    # Final pruning: keep only frequent (after merges)
    agg = {np_text: v for np_text, v in agg.items() if v["count"] >= min_keep}

    # Convert to requested NP_1, NP_2, ... format
    output = {}
    counter = 1

    for np_text, values in sorted(agg.items()):
        item = {
            "noun_phrase": np_text,
            "count": values["count"],
            "model_id": sorted(values["model_id"]),
            "sentence": sorted(values["sentence"]),
        }
        merged_from = sorted(values.get("merged_from") or [])
        if merged_from:
            item["merged_from"] = merged_from

        output[f"NP_{counter}"] = item
        counter += 1

    return output

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    Useful = {
    "Tool",
    # "Contributor",
    "Language Group",
    # "Locale",
    # "Language",
    "Model Component",
    "Infrastructure",
    "Dataset",
    "Benchmark",
    "License",
    # "Training Info",
    # "Metric",
}

    extractor = NounPhraseExtractor(
        NounPhraseExtractorConfig(
            spacy_model="en_core_web_sm",
            enable_validity_filter=True,
            enable_structure_filter=True,
            enable_concept_filter=True,
            enable_english_filter=True,
            enable_multiword_filter=False,
            useful_concepts=Useful,
            require_pass_count=None,          # all enabled must pass
            entity_match_mode="contain",      # "contain" or "overlap"
        )
    )

    # --- Single text extraction ---
    # sample = "This model uses English and French datasets under the Apache License."
    # print(extractor.extract_from_text(sample))

    # --- Folder processing (NP-relative entities) ---
    # LONG (recommended)
    # df_long = extractor.process_folder_np_relative_entities_df(
    #     directory="HF-Models-T6",
    #     output="long",
    #     include_only_passing_nps=True,
    #     save_path="2-NP_EXTRACTION/NP_global.csv",
    # )
    # print(df_long.head())

    # WIDE (one row per file)
    # df_wide = extractor.process_folder_np_relative_entities_df(
    #     directory="HF-Models-P9",
    #     output="wide",
    #     include_only_passing_nps=True,
    #     save_path="np_relative_entities_wide.parquet",
    # )
    # print(df_wide.head())

    np_dict = build_global_np_dictionary(
        extractor,
        directory="HF-Models-T6",
    )

    print(len(np_dict))
    print(np_dict["NP_1"])

    output_path = "2-NP_EXTRACTION/NP_global_dictionary_3.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(np_dict, f, indent=2, ensure_ascii=False)
