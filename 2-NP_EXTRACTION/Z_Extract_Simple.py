"""
Hierarchical NP storage + entity filtering ONLY on the entity parts.

Goal:
- Keep the full noun phrase node (e.g., "English and French datasets")
- Also store sub-spans inside it (entities like "English", "French") as children
- Apply "useful concept" filtering ONLY to those entity-children (not to the whole NP)
- Optionally require that an NP has at least one useful entity-child to be kept
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import re

import spacy
import emoji
from spacy.lang.en.stop_words import STOP_WORDS

from Ab_generic_terms import GENERIC_TERMS, EXCLUDE, WEAK_HEADS


ENTITY_TO_CONCEPT = {
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

DEFAULT_USEFUL_CONCEPTS = {
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


@dataclass
class HierNPConfig:
    spacy_model: str = "en_core_web_sm"

    # base NP filters (apply to full noun chunk)
    enable_validity_filter: bool = True
    enable_structure_filter: bool = True
    enable_english_filter: bool = True

    # entity filtering (apply ONLY to entity children inside NP)
    enable_entity_child_filter: bool = True
    useful_concepts: Set[str] = field(default_factory=lambda: set(DEFAULT_USEFUL_CONCEPTS))

    # if True: keep an NP only if it contains >=1 useful entity-child
    require_useful_entity_child: bool = False

    # if True: keep only multiword noun chunks as parents
    parent_multiword_only: bool = False


class HierarchicalNounPhraseExtractor:
    def __init__(self, config: Optional[HierNPConfig] = None):
        self.cfg = config or HierNPConfig()
        self.nlp = spacy.load(self.cfg.spacy_model)

    # -----------------------------
    # Parent NP filters (your logic)
    # -----------------------------
    def is_valid_np(self, np_text: str) -> bool:
        t = np_text.strip().lower()
        if re.fullmatch(r"\W+", t):
            return False
        if t in STOP_WORDS:
            return False
        if t in GENERIC_TERMS:
            return False
        if len(t.split()) == 1 and len(t) < 3:
            return False
        if t.isdigit():
            return False
        if any(t.startswith(det + " ") for det in {"the", "this", "that", "these", "those", "a", "an"}):
            return False
        if any(t.startswith(p.lower()) for p in EXCLUDE):
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

    def is_emoji_ok(self, sentence: str) -> bool:
        return emoji.emoji_count(sentence) == 0

    def process_emojis(self, sentence: str) -> str:
        return emoji.replace_emoji(sentence, "(emoji)")

    def is_english_like(self, text: str) -> bool:
        if any(ch in emoji.EMOJI_DATA for ch in text):
            return False
        for tok in text.split():
            tok_clean = re.sub(r"\W+", "", tok)
            if not tok_clean:
                continue
            if tok_clean.isalpha() and not LATIN_REGEX.match(tok_clean):
                return False
        return True

    # -----------------------------------------
    # Hierarchy building: head + modifiers + entities
    # -----------------------------------------
    def _extract_modifier_spans(self, chunk) -> List[Dict[str, Any]]:
        """
        Returns modifier “children” (non-entity) based on dependency relations.
        This is lightweight and meant for hierarchy/interpretability.
        """
        head = chunk.root
        children = []

        # Common modifier deps in NPs: amod, compound, nummod, poss, appos, etc.
        # We'll keep it focused to avoid noise.
        MOD_DEPS = {"amod", "compound", "nummod"}

        # Get tokens in the chunk that modify the head
        mods = [t for t in chunk if t.dep_ in MOD_DEPS and t.head == head]

        # Expand coordination: English and French -> include French via conjuncts
        expanded = []
        for m in mods:
            expanded.append(m)
            expanded.extend(list(m.conjuncts))
        # unique by token index
        expanded = {t.i: t for t in expanded}.values()

        for t in sorted(expanded, key=lambda x: x.i):
            children.append(
                {
                    "type": "modifier",
                    "text": t.text,
                    "lemma": t.lemma_,
                    "pos": t.pos_,
                    "dep": t.dep_,
                    "token_i": t.i,
                }
            )

        return children

    def _extract_entity_children(self, doc, chunk) -> List[Dict[str, Any]]:
        """
        Entity-children are entities that overlap the noun chunk span.
        This fixes the exact-span-match limitation and allows you to
        apply entity filtering on the relevant sub-span (e.g., "English").
        """
        ent_children = []
        for ent in doc.ents:
            # overlap test: [ent.start, ent.end) intersects [chunk.start, chunk.end)
            if ent.start < chunk.end and ent.end > chunk.start:
                concept = map_entity_to_concept(ent.label_)
                ent_children.append(
                    {
                        "type": "entity",
                        "text": ent.text,
                        "start": ent.start,
                        "end": ent.end,
                        "entity_label": ent.label_,
                        "concept": concept,
                        "is_useful": (concept in self.cfg.useful_concepts),
                    }
                )

        # dedupe exact duplicates (same span/label)
        seen = set()
        unique = []
        for c in ent_children:
            key = (c["start"], c["end"], c["entity_label"])
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    # -----------------------------------------
    # Main: produce hierarchical noun phrase nodes
    # -----------------------------------------
    def extract_hierarchical(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text or "")
        results: List[Dict[str, Any]] = []

        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            if not phrase:
                continue

            if self.cfg.parent_multiword_only and len(phrase.split()) <= 1:
                continue

            # Parent filters apply to the full chunk
            if self.cfg.enable_validity_filter and not self.is_valid_np(phrase):
                continue
            if self.cfg.enable_structure_filter and not self.is_structurally_useful_np(chunk):
                continue
            if self.cfg.enable_english_filter and not self.is_english_like(phrase):
                continue

            sent = chunk.sent.text.strip()
            if not self.is_emoji_ok(sent):
                sent = self.process_emojis(sent)

            node: Dict[str, Any] = {
                "type": "np",
                "text": phrase,
                "sentence": sent,
                "span": {"start": chunk.start, "end": chunk.end},
                "head": {
                    "text": chunk.root.text,
                    "lemma": chunk.root.lemma_,
                    "pos": chunk.root.pos_,
                    "token_i": chunk.root.i,
                },
                "children": [],
                # You can store additional diagnostics here
                "meta": {
                    "position_in_text": round((chunk.start / len(doc)) if len(doc) else 0.0, 3),
                },
            }

            # Add modifier children (no filtering needed)
            node["children"].extend(self._extract_modifier_spans(chunk))

            # Add entity children (filtering applies ONLY here)
            entity_children = self._extract_entity_children(doc, chunk)
            if self.cfg.enable_entity_child_filter:
                # Keep all entity children, but mark useful; you can also drop non-useful if you want.
                node["children"].extend(entity_children)
            else:
                # still store them (optional). If you truly want off, comment next line.
                node["children"].extend(entity_children)

            # Optionally require at least one useful entity-child to keep the parent NP
            if self.cfg.require_useful_entity_child:
                has_useful = any(
                    c.get("type") == "entity" and c.get("is_useful") is True
                    for c in node["children"]
                )
                if not has_useful:
                    continue

            results.append(node)

        return results


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":

    USEFUL_CONCEPTS = {"Tool", "Contributor", "Language Group", "Locale", "Language", "Model Component", "Infrastructure", "Dataset", "Benchmark", "License", "Training Info", "Metric",
}
    extractor = HierarchicalNounPhraseExtractor(
        HierNPConfig(
            useful_concepts=USEFUL_CONCEPTS,
            enable_entity_child_filter=True,
            require_useful_entity_child=False,  # set True if you only want NPs containing useful entities
        )
    )


    txt = "Model Card for Beit Geometric Shapes Dataset Base ## Training Dataset - **Repository:** https://huggingface.co/datasets/0-ma/geometric-shapes ## Base Model - **Repository:** https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k ## Accuracy - Accuracy on dataset 0-ma/geometric-shapes [test] : 0.9998 # Loading and using the model import numpy as np from PIL import Image from transformers import AutoImageProcessor, AutoModelForImageClassification import requests labels = [ None, Circle, Triangle, Square, Pentagon, Hexagon ] images = [Image.open(requests.get(https://raw.githubusercontent.com/0-ma/geometric-shape-detector/main/input/exemple_circle.jpg, stream=True).raw), Image.open(requests.get(https://raw.githubusercontent.com/0-ma/geometric-shape-detector/main/input/exemple_pentagone.jpg, stream=True).raw)] feature_extractor = AutoImageProcessor.from_pretrained(0-ma/beit-geometric-shapes-base) model = AutoModelForImageClassification.from_pretrained(0-ma/beit-geometric-shapes-base) inputs = feature_extractor(images=images, return_tensors=pt) logits = model(**inputs)[logits].cpu().detach().numpy() predictions = np.argmax(logits, axis=1) predicted_labels = [labels[prediction] for prediction in predictions] print(predicted_labels) ## Model generation The model has been created using the train_shape_detector.py of the project from the project https://github.com/0-ma/geometric-shape-detector. No external code sources were used."
    out = extractor.extract_hierarchical(txt)

    # Print a compact view
    for np_node in out:
        print("\nNP:", np_node["text"])
        print("  head:", np_node["head"]["text"])
        ents = [c for c in np_node["children"] if c["type"] == "entity"]
        mods = [c for c in np_node["children"] if c["type"] == "modifier"]
        print("  modifiers:", [m["text"] for m in mods])
        print("  entities:", [(e["text"], e["entity_label"], e["concept"], e["is_useful"]) for e in ents])


    # 2) Folder -> DF
    # df = extractor.process_folder_to_df("HF-Models-P9", save_path="np_rows.parquet")
    # print(df.shape)

    # 3) Aggregate DF
    # agg_df = extractor.aggregate_noun_phrases_df(df, min_global_freq=10)
    # agg_df.to_parquet("np_aggregated.parquet", index=False)
