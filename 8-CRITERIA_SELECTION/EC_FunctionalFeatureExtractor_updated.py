from dataclasses import dataclass, field
from typing import List, Iterable, Set, Dict
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from Ab_generic_terms import GENERIC_TERMS, EXCLUDE, WEAK_HEADS

# === NOUN PHRASE EXTRACTOR ===


class NounPhraseExtractor:
    """
    Encapsulates all logic for extracting useful noun phrases from text.

    This version keeps the same public class/method structure, but adds:
    1. phrase normalization
    2. canonicalization of common variants/synonyms
    3. deduplication by canonical form while preserving output order
    """

    DETERMINERS = {"the", "this", "that", "these", "those", "a", "an"}
    LEADING_MODIFIERS = {
        "strong",
        "robust",
        "advanced",
        "efficient",
        "effective",
        "powerful",
        "high-quality",
        "high quality",
        "scalable",
        "reliable",
        "flexible",
        "general",
    }
    TRAILING_GENERIC_HEADS = {
        "capability",
        "capabilities",
        "ability",
        "abilities",
        "support",
        "systems",
        "system",
    }
    PHRASE_CANONICAL_MAP: Dict[str, str] = {
        "language understanding capabilities": "language understanding",
        "understanding capabilities": "language understanding",
        "language understanding capability": "language understanding",
        "contextual embeddings": "contextual token embeddings",
        "token embeddings": "contextual token embeddings",
        "contextual token embedding": "contextual token embeddings",
        "finetuning": "fine-tuning",
        "fine tuning": "fine-tuning",
    }

    def __init__(
        self,
        generic_terms: Iterable[str] = GENERIC_TERMS,
        exclude_prefixes: Iterable[str] = EXCLUDE,
        weak_heads: Iterable[str] = WEAK_HEADS,
        nlp=None,
    ):
        self.generic_terms: Set[str] = {t.lower() for t in generic_terms}
        self.exclude_prefixes: Set[str] = {p.lower() for p in exclude_prefixes}
        self.weak_heads: Set[str] = {h.lower() for h in weak_heads}
        # Load spaCy model once
        self.nlp = nlp or spacy.load("en_core_web_sm")

    # --- internal helpers ---

    def _normalize_phrase(self, phrase: str) -> str:
        """
        Normalize a surface phrase into a more stable feature form.
        """
        phrase = phrase.strip()
        if not phrase:
            return ""

        # Normalize whitespace / dashes
        phrase = re.sub(r"[\u2010\u2011\u2012\u2013\u2014]", "-", phrase)
        phrase = re.sub(r"\s+", " ", phrase)

        # Remove surrounding punctuation but keep internal hyphens/slashes
        phrase = phrase.strip(" \t\n\r.,;:!?()[]{}\"'")
        lower = phrase.lower()

        # Strip leading determiners instead of rejecting the phrase.
        lower = re.sub(
            rf"^(?:{'|'.join(sorted(self.DETERMINERS, key=len, reverse=True))})\s+",
            "",
            lower,
        )

        # Strip leading promotional/evaluative modifiers.
        modifier_pattern = "|".join(
            sorted((re.escape(m) for m in self.LEADING_MODIFIERS), key=len, reverse=True)
        )
        lower = re.sub(rf"^(?:{modifier_pattern})\s+", "", lower)

        # Collapse "X capabilities/ability/support" -> "X"
        trailing_head_pattern = "|".join(
            sorted((re.escape(h) for h in self.TRAILING_GENERIC_HEADS), key=len, reverse=True)
        )
        lower = re.sub(rf"\s+(?:{trailing_head_pattern})$", "", lower)

        # Normalize common spelling / punctuation variants.
        lower = lower.replace("finetuning", "fine-tuning")
        lower = re.sub(r"\bfine tuning\b", "fine-tuning", lower)
        lower = re.sub(r"\s+", " ", lower).strip(" -_/.,;:!?")
        return lower

    def _canonicalize_phrase(self, phrase: str) -> str:
        """
        Map normalized variants to a canonical feature label.
        """
        phrase = self._normalize_phrase(phrase)
        if not phrase:
            return ""

        if phrase in self.PHRASE_CANONICAL_MAP:
            return self.PHRASE_CANONICAL_MAP[phrase]

        # Canonicalize a few common relational patterns.
        match = re.fullmatch(r"(.+?)\s+capabilit(?:y|ies)", phrase)
        if match:
            return match.group(1).strip()

        match = re.fullmatch(r"support for\s+(.+)", phrase)
        if match:
            return match.group(1).strip()

        return phrase

    def _format_feature_label(self, canonical_phrase: str, chunk) -> str:
        """
        Keep output human-readable while preserving canonical content.
        """
        if not canonical_phrase:
            return ""

        # Preserve acronyms / proper-noun casing from the original chunk when possible.
        original_tokens = [t.text for t in chunk if not t.is_space]
        original_lower_to_text = {}
        for token_text in original_tokens:
            token_key = token_text.lower()
            if token_key not in original_lower_to_text:
                original_lower_to_text[token_key] = token_text

        formatted_tokens = []
        for token in canonical_phrase.split():
            formatted_tokens.append(original_lower_to_text.get(token, token))

        return " ".join(formatted_tokens)

    def _is_valid_np(self, phrase: str) -> bool:
        phrase = self._canonicalize_phrase(phrase)
        # Only punctuation / empty after normalization
        if not phrase or re.fullmatch(r"\W+", phrase):
            return False
        # Just a stopword like "this", "those"
        if phrase in STOP_WORDS:
            return False
        # Generic term filter
        if phrase in self.generic_terms:
            return False
        # Single very short word
        if len(phrase.split()) == 1 and len(phrase) < 3:
            return False
        # Just a number
        if re.fullmatch(r"\d+(?:\.\d+)?", phrase):
            return False
        # Starts with excluded prefix
        if any(phrase.startswith(p) for p in self.exclude_prefixes):
            return False
        return True

    def _is_structurally_useful_np(self, chunk) -> bool:
        # Rule 1: all tokens are ADJ/ADV
        if all(token.pos_ in {"ADJ", "ADV"} for token in chunk):
            return False

        # Rule 2: adjective modifying another adjective
        for token in chunk:
            if token.dep_ == "amod" and token.head.pos_ == "ADJ":
                return False

        # Rule 3: very short NP with weak head
        if len(chunk) <= 2 and chunk.root.lemma_.lower() in self.weak_heads:
            return False

        # Must contain at least one noun-ish token
        if not any(token.pos_ in {"NOUN", "PROPN", "NUM"} for token in chunk):
            return False

        return True

    # --- public API ---

    def extract(self, text: str) -> List[str]:
        """
        Return a list of useful noun phrases found in `text`.

        Output is deduplicated by canonical form while preserving first occurrence.
        """
        doc = self.nlp(text)
        phrases: List[str] = []
        seen_canonical: Set[str] = set()

        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            canonical_phrase = self._canonicalize_phrase(phrase)

            if not self._is_valid_np(canonical_phrase):
                continue
            if not self._is_structurally_useful_np(chunk):
                continue
            if canonical_phrase in seen_canonical:
                continue

            seen_canonical.add(canonical_phrase)
            phrases.append(self._format_feature_label(canonical_phrase, chunk))

        return phrases


# === FUNCTIONAL FEATURES DATACLASS ===


@dataclass
class FunctionalFeatures:
    # Use default_factory to avoid mutable-default issues
    F_features: List[str] = field(default_factory=list)

    def add_features(self, features: List[str]) -> None:
        """
        Add a list of features to this object.
        """
        self.F_features.extend(features)

    def add_from_query(self, query: str, extractor: NounPhraseExtractor) -> None:
        """
        Use a NounPhraseExtractor to extract noun phrases from a query
        and store them as functional features.
        """
        noun_phrases = extractor.extract(query)
        self.add_features(noun_phrases)

    @classmethod
    def from_query(cls, query: str, extractor: NounPhraseExtractor) -> "FunctionalFeatures":
        """
        Convenience constructor: build a FunctionalFeatures instance
        directly from a query.
        """
        noun_phrases = extractor.extract(query)
        return cls(F_features=noun_phrases)


# === EXAMPLE USAGE ===

if __name__ == "__main__":
    extractor = NounPhraseExtractor()

    # query = "I need a multilingual encoder–decoder model to generate synthetic code-switched text by translating spans between Arabic and other languages, supporting many-to-many translation for data augmentation and enabling fine-tuning for downstream tasks."
    query = "I want a pretrained Arabic sequence-to-sequence model for generating short, fluent news titles from full articles. The model should be optimized for Arabic generation rather than multilingual coverage."
    # Option 1: start empty and add later
    ff = FunctionalFeatures()
    ff.add_from_query(query, extractor)
    print("F_features (option 1):", ff.F_features)

    # Option 2: build directly from the query
    ff2 = FunctionalFeatures.from_query(query, extractor)
    print("F_features (option 2):", ff2.F_features)
