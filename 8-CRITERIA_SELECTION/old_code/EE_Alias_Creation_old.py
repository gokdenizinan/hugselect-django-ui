from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import re
import json
import time

@dataclass(frozen=True)
class SynonymAlias:
    value: str
    confidence: float  # 0..1


class AliasResolver:
    """
    Produces two alias types:
      - grammatical aliases (safe, deterministic)
      - synonym aliases (semantic, risky) via an injected provider (LLM client, offline dict, etc.)

    Caches results per (feature_name, raw_value).
    """

    def __init__(
        self,
        *,
        synonym_provider: Optional[Callable[[str, str], List[SynonymAlias]]] = None,
        max_synonyms: int = 5,
        cache: Optional[Dict[Tuple[str, str], Tuple[List[str], List[SynonymAlias]]]] = None,
    ):
        self.synonym_provider = synonym_provider
        self.max_synonyms = max_synonyms
        self._cache = cache if cache is not None else {}

    # ---------- grammatical aliases ----------

    def grammatical_aliases(self, s: str) -> List[str]:
        if not s:
            return []

        raw = s.strip()
        # Keep original
        variants = {raw}

        # Lower / upper / title
        variants.add(raw.lower())
        variants.add(raw.upper())
        variants.add(raw.title())

        # Normalize whitespace
        collapsed_ws = re.sub(r"\s+", " ", raw).strip()
        variants.add(collapsed_ws)
        variants.add(collapsed_ws.lower())

        # Convert separators
        # space/underscore/dot -> dash
        dash = re.sub(r"[ _\.]+", "-", collapsed_ws).strip("-")
        variants.add(dash)
        variants.add(dash.lower())

        # dash -> space / underscore
        space = dash.replace("-", " ")
        variants.add(space)
        variants.add(space.lower())

        underscore = dash.replace("-", "_")
        variants.add(underscore)
        variants.add(underscore.lower())

        # Remove non-alnum except separators (keeps basic punctuation handling)
        cleaned = re.sub(r"[^a-zA-Z0-9\-_ ]+", "", collapsed_ws)
        if cleaned and cleaned != collapsed_ws:
            variants.add(cleaned)
            variants.add(cleaned.lower())
            variants.add(re.sub(r"[ _\.]+", "-", cleaned).strip("-").lower())

        # Compact CamelCase-ish: "TextClassification" -> "textclassification"
        compact = re.sub(r"[^a-zA-Z0-9]+", "", raw)
        if compact:
            variants.add(compact)
            variants.add(compact.lower())

        # Deduplicate and keep reasonable length
        out = [v for v in variants if v and len(v) <= 128]
        return sorted(set(out))

    # ---------- synonym aliases ----------

    def synonym_aliases(self, s: str) -> List[SynonymAlias]:
        if not self.synonym_provider:
            return []
        try:
            syns = self.synonym_provider( s) or []
        except Exception:
            return []
        # Sort + cap
        syns = sorted(syns, key=lambda a: a.confidence, reverse=True)[: self.max_synonyms]
        # Remove empties / self-duplicates
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

    # ---------- combined ----------
    def resolve(self, feature_name: str, raw_value: str) -> Tuple[List[str], List[SynonymAlias]]:
        key = (feature_name, raw_value)
        if key in self._cache:
            return self._cache[key]

        grams = self.grammatical_aliases(raw_value)
        syns = self.synonym_aliases( raw_value)

        self._cache[key] = (grams, syns)
        return grams, syns
    
    def resolve_syns(self, feature_name: str, raw_value: str,) -> List[SynonymAlias]:
        key = (feature_name, "syns", raw_value)

        if key in self._cache:
            return self._cache[key]

        syns = self.synonym_aliases(raw_value) or []

        self._cache[key] = syns
        return syns

    def resolve_grams(self, feature_name: str, raw_value: str,) -> List[SynonymAlias]:
        key = (feature_name, "grams", raw_value)

        if key in self._cache:
            return self._cache[key]

        grams = self.grammatical_aliases(raw_value) or []

        self._cache[key] = grams
        return grams
    
def stub_synonym_provider( value: str) -> List[SynonymAlias]:
    return []

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

# ---- cosine similarity ----

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b)))

# ---- embedding-based synonym provider ----

class EmbeddingSynonymProvider:
    def __init__(
        self,
        feature: str,
        model_name: str = "all-MiniLM-L6-v2",
        min_similarity: float = 0.1,
    ):
        self.model = SentenceTransformer(model_name)
        self.candidates = self.get_candidates(feature)
        self.min_similarity = min_similarity

        # Pre-embed candidate vocabulary once
        self.candidate_embeddings = self.model.encode(self.candidates,normalize_embeddings=True,)
    def get_candidates(self, feature : str):
        with open(f"8-CRITERIA_SELECTION/alias_candidates/{feature}.json", "r", encoding="utf-8") as f:
            CANDIDATES = json.load(f)
        return CANDIDATES
    
    def __call__(self, value: str):
        if not value.strip():
            return []

        # Embed input
        query_embedding = self.model.encode(
            value,
            normalize_embeddings=True,
        )

        results = []

        for candidate, candidate_emb in zip(self.candidates, self.candidate_embeddings):
            score = float(np.dot(query_embedding, candidate_emb))  # cosine similarity

            if score >= self.min_similarity:
                results.append(
                    SynonymAlias(
                        value=candidate,
                        confidence=score,
                    )
                )

        return results

def get_aliases(feature = str, value = str):
    provider = EmbeddingSynonymProvider(
    feature= feature,
    min_similarity=0.05,
    )

    resolver = AliasResolver(
        synonym_provider=provider,
        max_synonyms=10,
    )

    grams, syns = resolver.resolve(
    feature_name= feature,
    raw_value=value)
    return grams, syns

if __name__ == "__main__":
    start = time.perf_counter()
    print("==== ====")
    feature = "pipeline_tag"
    
    provider = EmbeddingSynonymProvider(
    feature= feature,
    min_similarity=0.05,
    )

    resolver = AliasResolver(
        synonym_provider=provider,
        max_synonyms=10,
    )

    grams = resolver.resolve_grams(
    feature_name= "pipeline_tag",
    raw_value="CODE, image generation",)
    print(grams)
    print("==== ====")
    syns = resolver.resolve_syns(
    feature_name= "pipeline_tag",
    raw_value="CODE",)

    print(syns)

    end = time.perf_counter()
    print(f"Time taken: {end - start:.3f} seconds")
    # values = [obj.value for obj in syns]
    # print(values) 
    # confidence = [obj.confidence for obj in syns]
    # print(confidence) 
    # print("")
    # gram , syn = get_aliases("pipeline_tag", "image genretion")
    # print(gram, syn)
    # print(syn[0].value)