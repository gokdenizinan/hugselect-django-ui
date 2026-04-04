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
from curses import raw
import re
from difflib import SequenceMatcher


import os
from tqdm import tqdm
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from collections import Counter
from rapidfuzz import process, fuzz

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

# --- Add this near the top of your first script (imports + preprocessing helpers) ---

import re
from typing import Tuple, Set, List

# Preprocessing regexes (ported from the second script)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]+`")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")

STACKTRACE_HINT_RE = re.compile(
    r"^\s*(Traceback|File \"|at \w+\.|Exception:|ERROR|WARN|INFO)\b", re.IGNORECASE
)
TABLEISH_LINE_RE = re.compile(r"^(\s*\|.*\|\s*|\s*[-+]{3,}\s*|\s*\d+(\s+\d+){3,}\s*)$")
MANY_NUMBERS_RE = re.compile(r"(\d[\d\.,%]*){6,}")
JSONISH_RE = re.compile(r"^\s*[\{\[]")
YAMLY_RE = re.compile(r"^\s*\w[\w\-]*:\s+.+")
SQLISH_RE = re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|WITH)\b", re.IGNORECASE)


def extract_code_blocks(text: str) -> List[str]:
    return CODE_FENCE_RE.findall(text or "")


def strip_code_blocks(text: str) -> str:
    return CODE_FENCE_RE.sub(" ", text or "")


def normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_noisy_lines(text: str, keep_short_lines: bool = True) -> str:
    out_lines = []
    for line in (text or "").splitlines():
        raw = line.rstrip("\n")
        s = raw.strip()

        if not s:
            out_lines.append("")
            continue

        if STACKTRACE_HINT_RE.search(s):
            continue
        if TABLEISH_LINE_RE.match(s):
            continue
        if MANY_NUMBERS_RE.search(s):
            continue
        if SQLISH_RE.match(s):
            continue

        if (JSONISH_RE.match(s) or YAMLY_RE.match(s)) and len(s) > 120:
            continue

        if not keep_short_lines and len(s) < 25 and not re.search(r"[A-Za-z]{6,}", s):
            continue

        out_lines.append(raw)

    return "\n".join(out_lines)


def preprocess_text(
    text: str,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_html: bool = True,
    remove_inline_code: bool = True,
    drop_noisy_lines: bool = True,
    keep_short_lines: bool = True,
) -> str:
    """
    Clean text for NP extraction.
    (This version returns only cleaned prose; if you also want code identifiers, tell me.)
    """
    cleaned = strip_code_blocks(text)

    if remove_inline_code:
        cleaned = INLINE_CODE_RE.sub(" ", cleaned)
    if remove_html:
        cleaned = HTML_TAG_RE.sub(" ", cleaned)
    if remove_urls:
        cleaned = URL_RE.sub(" ", cleaned)
    if remove_emails:
        cleaned = EMAIL_RE.sub(" ", cleaned)
    if drop_noisy_lines:
        cleaned = remove_noisy_lines(cleaned, keep_short_lines=keep_short_lines)

    cleaned = normalize_whitespace(cleaned)
    return cleaned


def map_entity_to_concept(label: str) -> str:
    return ENTITY_TO_CONCEPT.get(label, "Other")


def _mode(values: Iterable[Any]) -> Any:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    c = Counter(vals)
    return c.most_common(1)[0][0]

def _norm_np(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_np(a), _norm_np(b)).ratio()

LEADING_DEPS_TO_DROP = {"det", "poss"}  # "the", "a", "this", "my", etc.

def clean_np_from_chunk(chunk) -> str:
    tokens = []
    seen_content = False

    for tok in chunk:
        if not seen_content and tok.pos_ == "DET":
            continue
        seen_content = True
        tokens.append(tok.text)

    return " ".join(tokens)


LEADING_SYMBOL_RE = re.compile(r"^[\W_]+")  # non-word chars at beginning

def strip_leading_symbols(s: str) -> str:
    return LEADING_SYMBOL_RE.sub("", s).strip()

def normalize_np_from_chunk(chunk) -> str:
    text = clean_np_from_chunk(chunk)
    text = strip_leading_symbols(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------
# Config
# -------------------------
# --- Add preprocess config knobs to your existing dataclass ---

from dataclasses import dataclass, field
from typing import Optional, Set

@dataclass
class NounPhraseExtractorConfig:
    spacy_model: str = "en_core_web_sm"

    # Per-filter toggles (NP filtering)
    enable_validity_filter: bool = True
    enable_structure_filter: bool = True
    enable_concept_filter: bool = True
    enable_english_filter: bool = True
    enable_multiword_filter: bool = False

    useful_concepts: Set[str] = field(default_factory=lambda: set(DEFAULT_USEFUL_CONCEPTS))
    require_pass_count: Optional[int] = None
    entity_match_mode: str = "contain"

    # --- NEW: preprocessing toggles ---
    enable_preprocess: bool = True
    pp_remove_urls: bool = True
    pp_remove_emails: bool = True
    pp_remove_html: bool = True
    pp_remove_inline_code: bool = True
    pp_drop_noisy_lines: bool = True
    pp_keep_short_lines: bool = True



# -------------------------
# Extractor
# -------------------------
# --- Change your extractor to apply preprocess_text() inside extract_from_text() ---

class NounPhraseExtractor:
    def __init__(self, config: Optional[NounPhraseExtractorConfig] = None):
        self.config = config or NounPhraseExtractorConfig()
        self.nlp = spacy.load(self.config.spacy_model)
        self.reset_stats()

    def _maybe_preprocess(self, text: str) -> str:
        if not self.config.enable_preprocess:
            return text or ""
        return preprocess_text(
            text or "",
            remove_urls=self.config.pp_remove_urls,
            remove_emails=self.config.pp_remove_emails,
            remove_html=self.config.pp_remove_html,
            remove_inline_code=self.config.pp_remove_inline_code,
            drop_noisy_lines=self.config.pp_drop_noisy_lines,
            keep_short_lines=self.config.pp_keep_short_lines,
        )

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


    def is_valid_np(self,np_text: str) -> bool:
        raw = np_text.strip()
        lower = raw.lower()

        URL_REGEX = re.compile(r"https?://|www\.", re.IGNORECASE)

        if not raw:
            return False

        # Reject URLs
        if URL_REGEX.search(raw):
            return False

        # Reject only-symbol strings
        if re.fullmatch(r"\W+", raw):
            return False

        # Reject pure numbers
        if raw.isdigit():
            return False

        # NEW: reject spans that are mostly numbers
        # 1) character-based digit ratio (digits / (letters+digits))
        alnum = re.sub(r"[^A-Za-z0-9]+", "", raw)
        if alnum:
            digits = sum(ch.isdigit() for ch in alnum)
            digit_ratio = digits / len(alnum)
            if digit_ratio >= 0.60:   # tweak threshold as needed
                return False

        # 2) token-based numeric ratio (numeric tokens / total tokens)
        tokens = re.findall(r"[A-Za-z0-9]+", raw)
        if tokens:
            num_tokens = sum(t.isdigit() for t in tokens)
            if (num_tokens / len(tokens)) >= 0.60:
                return False

        # Stopwords / generic terms (usually best to compare lowercased)
        if lower in STOP_WORDS:
            return False
        if lower in GENERIC_TERMS:
            return False

        # Very short single token
        if len(raw.split()) == 1 and len(raw) < 3:
            return False

        if any(lower.startswith(p.lower()) for p in EXCLUDE):
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
        # NEW: clean first
        text = self._maybe_preprocess(text)

        doc = self.nlp(text or "")


        # Keep this if you still want it for debugging/inspection, but it's no longer used for labeling.
        ent_span_to_label = {(ent.start, ent.end): ent.label_ for ent in doc.ents}

        key_entity_labels = set(ENTITY_TO_CONCEPT.keys())
        required = self._required_passes()
        entries: List[Dict[str, Any]] = []

        # --- helpers (define once) ---
        def label_chunk_if_contains_entity(chunk, doc) -> str:
            for ent in doc.ents:
                if chunk.start <= ent.start and chunk.end >= ent.end:
                    return ent.label_
            return "O"

        def label_chunk_by_best_overlap(chunk, doc) -> str:
            """Return the label of the entity with the largest token-overlap with chunk, else 'O'."""
            best_ent = None
            best_overlap = 0

            for ent in doc.ents:
                overlap = min(chunk.end, ent.end) - max(chunk.start, ent.start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_ent = ent

            return best_ent.label_ if best_overlap > 0 else "O"

        def keep_chunk(chunk) -> bool:
            """Reject junky chunks (pronouns, all-stopword/punct)."""
            # single-token pronouns: This, It, us, etc.
            if len(chunk) == 1 and chunk.root.pos_ == "PRON":
                return False

            # spans that are entirely stop/punct/space/symbol
            if all(t.is_stop or t.is_punct or t.is_space or t.pos_ == "SYM" for t in chunk):
                return False

            # very short symbol-ish strings after stripping
            s = chunk.text.strip()
            if not s or s in {"*", "#"}:
                return False

            return True

        # --- main loop ---
        for chunk in doc.noun_chunks:
            self.filter_counts["Total"] += 1

            if not keep_chunk(chunk):
                continue

            phrase = normalize_np_from_chunk(chunk)
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

            # --- Concept labeling: OVERLAP-based (replaces exact span lookup) ---
            entity_label = label_chunk_by_best_overlap(chunk, doc)
            # entity_label = label_chunk_if_contains_entity(chunk, doc)

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

        long_rows: List[Dict[str, Any]] = []
        wide_rows: List[Dict[str, Any]] = []

        files = sorted(f for f in os.listdir(directory) if f.endswith(file_suffix))

        for filename in files:
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            model_id = data.get("modelId", filename.replace(file_suffix, ""))
            text = data.get(description_key, "") or ""

            # preprocess once
            text_clean = self._maybe_preprocess(text)
            doc = self.nlp(text_clean)

            # Decide which NPs to keep (must be computed on the SAME cleaned text)
            if include_only_passing_nps:
                kept_entries = self.extract_from_text(text_clean)
                kept_set = {e["noun_phrase"] for e in kept_entries}
            else:
                kept_set = None

            # Initialize per-file accumulators (YOU NEED THESE)
            np_texts: List[str] = []
            heads: List[str] = []
            modifiers_all: List[List[str]] = []
            ents_all: List[List[str]] = []
            ent_labels_all: List[List[str]] = []
            ent_concepts_all: List[List[str]] = []

            file_entity_seen: set[Tuple[int, int, str]] = set()

            for chunk in doc.noun_chunks:
                np_text = normalize_np_from_chunk(chunk)
                # np_text = chunk.text.strip()
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

from collections import defaultdict
import os, json

def build_global_np_dictionary(
    extractor,
    directory: str,
    description_key: str = "description",
    file_suffix: str = ".json",
):
    agg = defaultdict(lambda: {
        "model_id": set(),
        "sentence": set(),
        "entity_labels": set(),   # NEW
        "count": 0,
    })

    files = sorted(f for f in os.listdir(directory) if f.endswith(file_suffix))

    sample_fraction = 1  # 1%
    sample_size = max(1, int(len(files) * sample_fraction))
    files = files[:sample_size]

    for filename in tqdm(files, desc="Processing files", unit="file"):
        file_path = os.path.join(directory, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        model_id = data.get("modelId", filename.replace(file_suffix, ""))
        text = data.get(description_key, "") or ""

        # NEW: preprocess before NP extraction
        text_clean = extractor._maybe_preprocess(text)

        np_entries = extractor.extract_from_text(text_clean)

        for entry in np_entries:
            np_text = entry["noun_phrase"]
            sentence = entry["sentence"]

            agg[np_text]["count"] += 1
            agg[np_text]["model_id"].add(model_id)
            agg[np_text]["sentence"].add(sentence)

            # NEW: collect labels (skip "O" if you don't want non-entities)
            lbl = entry.get("entity_label")
            if lbl and lbl != "O":
                agg[np_text]["entity_labels"].add(lbl)

    min_keep = 5
    sim_threshold = 90  # RapidFuzz returns 0..100

    # --- precompute normalized forms ONCE ---
    norm_cache = {}
    def norm(s: str) -> str:
        s = s or ""
        try:
            return norm_cache[s]
        except KeyError:
            ns = _norm_np(s)
            norm_cache[s] = ns
            return ns

    frequent = [np_text for np_text, v in agg.items() if v["count"] >= min_keep]

    # --- better blocking than "first word": use 3-gram prefix or first 2 tokens ---
    bucket = {}
    for f in frequent:
        nf = norm(f)
        key = " ".join(nf.split()[:2])  # first 2 tokens tends to work better
        bucket.setdefault(key, []).append(f)

    frequent_set = set(frequent)

    # optional: pre-normalize frequent candidates for faster matching
    # RapidFuzz can also apply a processor per string; we’ll pass normalized query and normalized candidates instead.
    frequent_norm = {f: norm(f) for f in frequent}

    for np_text, v in list(agg.items()):
        if v["count"] >= min_keep:
            continue

        n = norm(np_text)
        key = " ".join(n.split()[:2])
        candidates = bucket.get(key)
        if not candidates:
            # fallback: if no bucket, you can skip or use all frequent (but that can be slow)
            candidates = frequent

        # match against *normalized* candidates
        # build list of normalized strings once per candidates list
        cand_norms = [frequent_norm[c] for c in candidates]

        match = process.extractOne(
            n,
            cand_norms,
            scorer=fuzz.ratio,
            score_cutoff=sim_threshold,
        )

        if match is None:
            continue

        # match returns (matched_string, score, index)
        _, score, idx = match
        best_match = candidates[idx]

        # merge
        agg[best_match]["count"] += v["count"]
        agg[best_match]["model_id"].update(v["model_id"])
        agg[best_match]["sentence"].update(v["sentence"])
        agg[best_match]["entity_labels"].update(v["entity_labels"])
        agg[best_match].setdefault("merged_from", set()).add(np_text)
        del agg[np_text]

    agg = {np_text: v for np_text, v in agg.items() if v["count"] >= min_keep}

    out = {}
    i = 1
    for np_text, v in agg.items():
        out[f"NP_{i}"] = {
            "noun_phrase": np_text,
            "count": v["count"],
            "model_id": sorted(v["model_id"]),
            "sentence": sorted(v["sentence"]),
            "entity_labels": sorted(v["entity_labels"]),  # NEW
        }
        if "merged_from" in v:
            out[f"NP_{i}"]["merged_from"] = sorted(v["merged_from"])
        i += 1

    return out



def print_noun_phrases_and_entities(noun_phrase_data):
    for item in noun_phrase_data:
        noun_phrase = item.get("noun_phrase")
        entity_label = item.get("entity_label")
        print(f"{noun_phrase} ({entity_label})")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    Useful = {
    "Tool",
    "Language Group",
    "Model Component",
    "Infrastructure",
    "Dataset",
    "Benchmark",
    "License",
    # "Training Info",
    # "Metric",
    # "Locale",
    # "Language",
    # "Contributor",

}

    extractor = NounPhraseExtractor(
        NounPhraseExtractorConfig(
            spacy_model="en_core_web_sm",
            enable_preprocess=True,
            pp_remove_inline_code=False,
            pp_drop_noisy_lines=False,
            enable_validity_filter=True,
            enable_structure_filter=True,
            enable_concept_filter=False,
            enable_english_filter=True,
            enable_multiword_filter=False,
            useful_concepts=Useful,
            require_pass_count=None,          # all enabled must pass
            entity_match_mode="contain",      # "contain" or "overlap"
        )
    )

    # --- Single text extraction ---
    # sample = "This model uses English and French datasets under the Apache License."
    # sample = "This is a strong pre-trained RoBERTa-Large NLI model. The training data is a combination of well-known NLI datasets: SNLI, MNLI, FEVER-NLI, ANLI (R1, R2, R3). Other pre-trained NLI models including RoBERTa, ALBert, BART, ELECTRA, XLNet are also available. Trained by Yixin Nie, original source. Try the code snippet below. More in here. Citation:"
    sample = "RuBERT for NLI (natural language inference) This is the DeepPavlov/rubert-base-cased fine-tuned to predict the logical relationship between two short texts: entailment, contradiction, or neutral. ## Usage How to run the model for NLI: Alternatively, you can use Huggingface pipelines for inference. ## Sources The model has been trained on a series of NLI datasets automatically translated to Russian from English. Most datasets were taken from the repo of Felipe Salvatore: JOCI, MNLI, MPE, SICK, SNLI. Some datasets obtained from the original sources: ANLI, NLI-style FEVER, IMPPRES. ## Performance The table below shows ROC AUC (one class vs rest) for five models on the corresponding *dev* sets: - tiny: a small BERT predicting entailment vs not_entailment - twoway: a base-sized BERT predicting entailment vs not_entailment - threeway (**this model**): a base-sized BERT predicting entailment vs contradiction vs neutral - vicgalle-xlm: a large multilingual NLI model - facebook-bart: a large multilingual NLI model For evaluation (and for training of the tiny and twoway models), some extra datasets were used: Add-one RTE, CoPA, IIE, and SCITAIL taken from the repo of Felipe Salvatore and translatted, HELP and MoNLI taken from the original sources and translated, and Russian TERRa."
    # sample = "Fantasy Sword on Stable Diffusion via Dreambooth This the Stable Diffusion model fine-tuned the Fantasy Sword concept taught to Stable Diffusion with Dreambooth. It can be used by modifying the instance_prompt: **a photo of fantasy_sword** # Run on Mirage Run this model and explore text-to-3D on Mirage! Here are is a sample output for this model: !image 0 # Share your Results and Reach us on Discord! ![Discord Server](https://discord.gg/9B2Pu2bEvj) Image Source"
    # sample = "A small multilingual utility model intended for simple text correction. It is designed to improve the quality of texts from the web, often lacking punctuation or proper word capitalization. The model was trained to perform three types of corrections: * Restoring punctuation in sentences. * Restoring word capitalization. * Restoring diacritical marks for languages that include them. The following languages are supported: Belarusian (be), Danish (da), German (de), Greek (el), English (en), Spanish (es), French (fr), Italian (it), Dutch (nl), Polish (pl), Portuguese (pt), Romanian (ro), Russian (ru), Slovak (sk), Swedish (sv), Ukrainian (uk). The model takes as input a sentence preceded by a language code prefix. For example:"
    # sample = "This model is not designed for general sentiment analysis or other NLP tasks."
    # sample = "This model is a fine-tuned version of gpt2-medium on an unknown dataset. It achieves the following results on the evaluation set: - Train Loss: 2.2131 - Epoch: 4 ## Model description More information needed ## Intended uses & limitations More information needed ## Training and evaluation data More information needed ## Training procedure ### Training hyperparameters The following hyperparameters were used during training: - optimizer: {name: Adam, weight_decay: None, clipnorm: None, global_clipnorm: None, clipvalue: None, use_ema: False, ema_momentum: 0.99, ema_overwrite_frequency: None, jit_compile: True, is_legacy_optimizer: False, learning_rate: {module: transformers.optimization_tf, class_name: WarmUp, config: {initial_learning_rate: 5e-05, decay_schedule_fn: {module: keras.optimizers.schedules, class_name: PolynomialDecay, config: {initial_learning_rate: 5e-05, decay_steps: -730, end_learning_rate: 0.0, power: 1.0, cycle: False, name: None}, registered_name: None}, warmup_steps: 1000, power: 1.0, name: None}, registered_name: WarmUp}, beta_1: 0.9, beta_2: 0.999, epsilon: 1e-08, amsgrad: False} - training_precision: float32 ### Training results ### Framework versions - Transformers 4.35.2 - TensorFlow 2.15.0 - Datasets 2.16.0 - Tokenizers 0.15.0"
    # sample = "Fine-tuned French Voxpopuli wav2vec2 large model for speech recognition in French Fine-tuned facebook/wav2vec2-large-fr-voxpopuli on French using the train and validation splits of Common Voice 6.1. When using this model, make sure that your speech input is sampled at 16kHz. This model has been fine-tuned thanks to the GPU credits generously given by the OVHcloud :) The script used for training can be found here: https://github.com/jonatasgrosman/wav2vec2-sprint ## Usage The model can be used directly (without a language model) as follows... Using the HuggingSound library: Writing your own inference script: ## Evaluation The model can be evaluated as follows on the French (fr) test data of Common Voice. **Test Result**: In the table below I report the Word Error Rate (WER) and the Character Error Rate (CER) of the model. I ran the evaluation script described above on other models as well (on 2021-05-16). Note that the table below may show different results from those already reported, this may have been caused due to some specificity of the other evaluation scripts used. ## Citation If you want to cite this model you can use this:"
    # sample = "PPO** Agent playing **LunarLander-v0** This is a trained model of a **PPO** agent playing **LunarLander-v0** using the stable-baselines3 library. ## Usage (with Stable-baselines3) TODO: Add your code"
    print_noun_phrases_and_entities(extractor.extract_from_text(sample))
    # print(extractor.extract_from_text(sample))
# ['Stable Diffusion model', 'Fantasy Sword concept', 'Dreambooth', 'instance_prompt', 'photo', 'Mirage', 'text-to-3D', 'Discord Server', 'image']
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
    #     include_only_passing_nps=True,1
    #     save_path="np_relative_entities_wide.parquet",
    # )
    # print(df_wide.head())

    # --- Build global NP dictionary with merging ---
    # np_dict = build_global_np_dictionary(
    #     extractor,
    #     directory="HF-Models-T6",
    # )

    # print(len(np_dict))
    # print(np_dict["NP_1"])

    # output_path = "2-NP_EXTRACTION/NP_global_dictionary_comfy.json"

    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(np_dict, f, indent=2, ensure_ascii=False)
 