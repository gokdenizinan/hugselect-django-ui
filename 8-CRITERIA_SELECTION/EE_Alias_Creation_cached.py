from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers.utils import logging
logging.set_verbosity_error()

import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class SynonymAlias:
    value: str
    confidence: float  # 0..1


SynonymProvider = Callable[[str], List[SynonymAlias]]
# NOTE: factory now takes List[str] (features), not a single str
SynonymProviderFactory = Callable[[List[str]], SynonymProvider]


_GLOBAL_MODELS: Dict[str, SentenceTransformer] = {}
_GLOBAL_CANDIDATES: Dict[Tuple[str, ...], List[str]] = {}
_GLOBAL_EMBEDDINGS: Dict[Tuple[str, Tuple[str, ...]], np.ndarray] = {}


def get_model(model_name: str) -> SentenceTransformer:
    model = _GLOBAL_MODELS.get(model_name)
    if model is None:
        print(f"Loading embedding model once: {model_name}")
        model = SentenceTransformer(model_name)
        _GLOBAL_MODELS[model_name] = model
    return model



class AliasResolver:
    def __init__(
        self,
        *,
        synonym_provider_factory: Optional[SynonymProviderFactory] = None,
        max_synonyms: int = 5,
    ):
        self.synonym_provider_factory = synonym_provider_factory
        self.max_synonyms = max_synonyms

        self._cache_grams: Dict[Tuple[str, str], List[str]] = {}
        self._cache_syns: Dict[Tuple[str, str], Tuple[List[str], List[float]]] = {}

        self._cache_syns_grouped: Dict[Tuple[str, str], Dict[str, List[Tuple[str, float]]]] = {}
        self._cache_grams_grouped: Dict[Tuple[str, str], Dict[str, List[str]]] = {}


        # cache providers per (features list as a stable key)
        self._provider_by_feature_key: Dict[Tuple[str, ...], SynonymProvider] = {}

    def grammatical_aliases(self, s: str) -> List[str]:
        if not s:
            return []
        raw = s.strip()
        variants = {raw}

        variants.add(raw.lower())
        variants.add(raw.upper())
        variants.add(raw.title())

        collapsed_ws = re.sub(r"\s+", " ", raw).strip()
        variants.add(collapsed_ws)
        variants.add(collapsed_ws.lower())

        dash = re.sub(r"[ _\.]+", "-", collapsed_ws).strip("-")
        variants.add(dash)
        variants.add(dash.lower())

        space = dash.replace("-", " ")
        variants.add(space)
        variants.add(space.lower())

        underscore = dash.replace("-", "_")
        variants.add(underscore)
        variants.add(underscore.lower())

        cleaned = re.sub(r"[^a-zA-Z0-9\-_ ]+", "", collapsed_ws)
        if cleaned and cleaned != collapsed_ws:
            variants.add(cleaned)
            variants.add(cleaned.lower())
            variants.add(re.sub(r"[ _\.]+", "-", cleaned).strip("-").lower())

        compact = re.sub(r"[^a-zA-Z0-9]+", "", raw)
        if compact:
            variants.add(compact)
            variants.add(compact.lower())

        out = [v for v in variants if v and len(v) <= 128]
        return sorted(set(out))

    def _get_provider(self, features: List[str]) -> Optional[SynonymProvider]:
        if not self.synonym_provider_factory:
            return None
        key = tuple(sorted(features))
        if key not in self._provider_by_feature_key:
            self._provider_by_feature_key[key] = self.synonym_provider_factory(features)
        return self._provider_by_feature_key[key]

    def synonym_aliases(self, provider: SynonymProvider, value: str) -> List[SynonymAlias]:
        if not value or not value.strip():
            return []
        try:
            syns = provider(value) or []
        except Exception:
            return []

        syns = sorted(syns, key=lambda a: a.confidence, reverse=True)[: self.max_synonyms]

        cleaned: List[SynonymAlias] = []
        seen = set()
        for a in syns:
            v = (a.value or "").strip()
            if not v:
                continue
            k = v.lower()
            if k in seen:
                continue
            seen.add(k)
            cleaned.append(SynonymAlias(value=v, confidence=max(0.0, min(1.0, float(a.confidence)))))
        return cleaned

    def resolve_syns(self, feature_names: List[str], raw_values: List[str]) -> Tuple[List[str], List[float]]:
        key = (str(feature_names), str(raw_values))
        if key in self._cache_syns:
            return self._cache_syns[key]

        provider = self._get_provider(feature_names)
        if not provider:
            return ([], [])

        all_syns: List[SynonymAlias] = []
        for raw in raw_values:
            all_syns.extend(self.synonym_aliases(provider, raw))

        syns_vals = [s.value for s in all_syns]
        syns_weights = [s.confidence for s in all_syns]

        self._cache_syns[key] = (syns_vals, syns_weights)
        return syns_vals, syns_weights

    def resolve_grams(self, feature_name: List[str], raw_values: List[str]) -> List[str]:
        key = (str(feature_name), str(raw_values))
        if key in self._cache_grams:
            return self._cache_grams[key]

        all_grams: List[str] = []
        for raw in raw_values:
            all_grams.extend(self.grammatical_aliases(raw))

        self._cache_grams[key] = all_grams
        return all_grams
    
    def resolve_syns_grouped(self, feature_names: List[str], raw_values: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Returns per raw value synonyms:
        { raw_value: [(syn, confidence), ...] }
        Guarantees synonyms are NOT mixed across values.
        """
        key = (str(feature_names), str(raw_values))
        if key in self._cache_syns_grouped:
            return self._cache_syns_grouped[key]

        provider = self._get_provider(feature_names)
        if not provider:
            out: Dict[str, List[Tuple[str, float]]] = {rv: [] for rv in raw_values}
            self._cache_syns_grouped[key] = out
            return out

        out: Dict[str, List[Tuple[str, float]]] = {}
        for raw in raw_values:
            aliases = self.synonym_aliases(provider, raw)  # already sorted, deduped, top-k
            out[raw] = [(a.value, float(a.confidence)) for a in aliases]

        self._cache_syns_grouped[key] = out
        return out

    def resolve_grams_grouped(self, feature_names: List[str], raw_values: List[str]) -> Dict[str, List[str]]:
        """
        Returns per raw value grams:
        { raw_value: [gram1, gram2, ...] }
        """
        key = (str(feature_names), str(raw_values))
        if key in self._cache_grams_grouped:
            return self._cache_grams_grouped[key]

        out: Dict[str, List[str]] = {}
        for raw in raw_values:
            out[raw] = self.grammatical_aliases(raw)

        self._cache_grams_grouped[key] = out
        return out


    def resolve(
        self,
        feature_name: List[str],
        raw_values: List[str],
    ) -> Tuple[List[str], Tuple[List[str], List[float]]]:
        grams = self.resolve_grams(feature_name, raw_values)
        syns = self.resolve_syns(feature_name, raw_values)
        return grams, syns

class EmbeddingSynonymProvider:
    def __init__(
        self,
        features: List[str],
        model_name: str = "all-MiniLM-L6-v2",
        min_similarity: float = 0.1,
    ):
        normalized_features = tuple(sorted(features))
        self.features = list(normalized_features)
        self.min_similarity = min_similarity
        self.model = get_model(model_name)

        candidate_key = normalized_features
        if candidate_key not in _GLOBAL_CANDIDATES:
            _GLOBAL_CANDIDATES[candidate_key] = self._get_candidates(self.features)
        self.candidates = _GLOBAL_CANDIDATES[candidate_key]

        embedding_key = (model_name, normalized_features)
        if embedding_key not in _GLOBAL_EMBEDDINGS:
            _GLOBAL_EMBEDDINGS[embedding_key] = self.model.encode(
                self.candidates,
                normalize_embeddings=True,
            )

        self.candidate_embeddings = _GLOBAL_EMBEDDINGS[embedding_key]

    def _get_candidates(self, features: List[str]) -> List[str]:
        candidates: List[str] = []
        for fe in features:
            with open(f"8-CRITERIA_SELECTION/alias_candidates/{fe}.json", "r", encoding="utf-8") as f:
                candidates.extend(json.load(f))
        return candidates

    def __call__(self, value: str) -> List[SynonymAlias]:
        if not value.strip():
            return []

        q = self.model.encode(value, normalize_embeddings=True)
        scores = self.candidate_embeddings @ q

        idx = np.where(scores >= self.min_similarity)[0]
        if idx.size == 0:
            return []

        idx = idx[np.argsort(scores[idx])[::-1]]
        return [SynonymAlias(value=self.candidates[i], confidence=float(scores[i])) for i in idx]


def make_embedding_provider_factory(
    *,
    model_name: str = "all-MiniLM-L6-v2",
    min_similarity: float = 0.05,
) -> SynonymProviderFactory:
    def factory(feature_names: List[str]) -> SynonymProvider:
        return EmbeddingSynonymProvider(
            features=feature_names,
            model_name=model_name,
            min_similarity=min_similarity,
        )
    return factory



if __name__ == "__main__":
    resolver = AliasResolver(
        synonym_provider_factory=make_embedding_provider_factory(min_similarity=0.2),
        max_synonyms=100,
    )
    syns, syns_weights = resolver.resolve_syns(
        ["Features"],
        ["many-to-many translation"],
    )
    print("SYNS:", syns)
    print("SYN WEIGHTS:", syns_weights)

    print("= = = = = = = = = = = =")
    grams = resolver.resolve_grams(
        ["language"],
        ["arabic"],
    )
    print("GRAMS:", grams)
