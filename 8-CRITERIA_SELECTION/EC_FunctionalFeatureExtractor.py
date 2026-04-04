
from dataclasses import dataclass, field
from typing import List, Optional, Iterable, Set
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from Ab_generic_terms import GENERIC_TERMS, EXCLUDE, WEAK_HEADS

# === NOUN PHRASE EXTRACTOR ===

class NounPhraseExtractor:
    """
    Encapsulates all logic for extracting useful noun phrases from text.
    """

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

    def _is_valid_np(self, phrase: str) -> bool:
        phrase = phrase.strip().lower()
        # Only punctuation
        if re.fullmatch(r"\W+", phrase):
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
        if phrase.isdigit():
            return False
        # Starts with a determiner
        if any(phrase.startswith(det + " ") for det in {"the", "this", "that", "these", "those", "a", "an"}):
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
        """
        doc = self.nlp(text)
        phrases: List[str] = []

        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()

            if not self._is_valid_np(phrase):
                continue
            if not self._is_structurally_useful_np(chunk):
                continue

            phrases.append(phrase)

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
    print("= = =")
    # query = "I want a pretrained Arabic sequence-to-sequence model designed for Arabic language understanding and fine tuning, capable of generating short, fluent news titles from full articles, with a focus on Arabic generation instead of multilingual coverage."
    print("original")
    query = "I need a Chinese large language model with solid instruction-following and safety characteristics to serve as a base for domain-specific medical adaptation. The model should support further pre-training and reinforcement learning."
    
    ff = FunctionalFeatures()
    ff.add_from_query(query, extractor)
    print("F_features (option 1):", ff.F_features)

    print("altered")  
    query = "I need a Chinese large language model with solid instruction-following, safety characteristics, and reasoning capabilities to serve as a base for domain-specific medical adaptation using Chinese data. The model should support further pre-training, SFT training, and Proximal Policy Optimization (PPO)."  
        # Option 1: start empty and add later
    ff.add_from_query(query, extractor)
    print("F_features (option 2):", ff.F_features)

    # # Option 2: build directly from the query
    # ff2 = FunctionalFeatures.from_query(query, extractor)
    # print("F_features (option 2):", ff2.F_features)