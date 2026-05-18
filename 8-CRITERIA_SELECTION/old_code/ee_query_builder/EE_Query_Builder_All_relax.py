from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal, Union, Iterable

from datetime import datetime, timedelta
from collections import defaultdict
import re
import os
import json

query_mapping = {
    # high-level intent
    "task_field": ["Metadata.pipeline_tag", "Metadata.tags", "Metadata.model_type", "Features"],
    "domain_field": ["Metadata.tags", "Features"],
    # Support both legacy and current item schemas.
    "model_name_field": ["model_id", "modelID"],
    "author_field": ["author"],
    # Objective-like evidence often appears only in free-text descriptions.
    "objective_field": ["Metadata.tags", "Features"],

    # metadata / constraints
    "license_field": ["Metadata.license", "Metadata.tags"],
    "downloads_all_time_field": "Metadata.downloads_all_time",
    "likes_field": "Metadata.likes",
    "downloads_30d_field": "Metadata.downloads_last_30_days",
    "file_count_field": "Metadata.file_count",
    "gated_field": "Metadata.gated",
    "library_name_field": ["Metadata.library_name", "Features"],
    "model_type_field": ["Metadata.model_type", "Features"],
    "basemodels_field": ["Metadata.basemodels", "Features"],
    "datasets_field": ["Metadata.datasets", "Features"],
    "tensors_total_field": "Metadata.tensors_total",
    "used_storage_field": "Metadata.usedStorage",
    "last_modified_field": "Metadata.lastModified",
    "language_field": ["Metadata.language", "Metadata.tags", "Features"],
    "metrics_field": ["Metadata.metrics", "Features"],

    # dedicated functional-feature search space
    "functional_search_fields": [
        "Features",
        "Metadata.tags",
        "Metadata.pipeline_tag",
        "Metadata.model_type",
        "Metadata.library_name",
        "Metadata.basemodels",
        "Metadata.datasets",
        "Metadata.language",
        "Metadata.metrics",
    ],

    # quality dimensions (numeric score fields)
    "functional_suitability_field": "Quality.Functional Suitability.score",
    "compatibility_field": "Quality.Compatibility.score",
    "performance_efficiency_field": "Quality.Performance Efficiency.score",
    "reliability_field": "Quality.Reliability.score",
    "interaction_capability_field": "Quality.Interaction Capability.score",
    "security_field": "Quality.Security.score",
    "maintainability_field": "Quality.Maintainability.score",
    "flexibility_field": "Quality.Flexibility.score",
}
# ----------------------------
# Types
# ----------------------------

PreferencePriority = Literal["must", "strong_prefer", "prefer", "avoid"]
RelaxLevel = Literal[0, 1, 2, 3, 4]  # 0 strict, 1 synonym-must, 2 should-gram, 3 should-syn, 4 dropped
MappingValue = Union[str, List[str]]
MappingType = Dict[str, MappingValue]

# External contracts (yours)
from EE_Alias_Creation import AliasResolver, make_embedding_provider_factory


# ----------------------------
# FeatureGroup: relax feature-by-feature with levels
# ----------------------------

@dataclass
class FeatureGroup:
    feature_key: str                         # e.g. "license_name", "domain", "author"
    priority: PreferencePriority              # must/strong_prefer/prefer/avoid
    # user inputs (normalized values)
    include: List[Any] = field(default_factory=list)
    exclude: List[Any] = field(default_factory=list)

    # fields (ES paths) that this feature can match on
    fields: List[str] = field(default_factory=list)

    # computed variants
    grams_by_value: Dict[str, List[str]] = field(default_factory=dict)
    syn_by_value: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)

    # relaxation state
    level: int = 0                           # 0..4
    relaxable: bool = True                   # task might be False, etc.

    # scoring
    base_weight: float = 1.0                 # feature-level base weight
    per_value_weight: Optional[float] = None # if you want per-value weights


# ----------------------------
# Helpers: ES query building primitives
# ----------------------------

def _emit_term(field: str, value: Any) -> Dict[str, Any]:
    return {"term": {field: value}}

def _emit_terms(field: str, values: List[Any]) -> Dict[str, Any]:
    return {"terms": {field: values}}

def _emit_range(field: str, body: Dict[str, Any]) -> Dict[str, Any]:
    return {"range": {field: body}}

def _wrap_constant_score(filter_clause: Dict[str, Any], boost: float) -> Dict[str, Any]:
    return {"constant_score": {"filter": filter_clause, "boost": float(boost)}}

def _bool_should(clauses: List[Dict[str, Any]], msm: int = 1) -> Dict[str, Any]:
    return {"bool": {"should": clauses, "minimum_should_match": int(msm)}}

def _terms_many(fields: List[str], values: List[str], k: int = 1) -> Dict[str, Any]:
    """
    Requires at least k distinct values to match; each value can match any of the fields.
    Useful when you want "k-of-n values".
    """
    per_value_groups = [
        {"bool": {"should": [{"term": {f: v}} for f in fields], "minimum_should_match": 1}}
        for v in values
    ]
    return {"bool": {"should": per_value_groups, "minimum_should_match": int(k)}}

def _terms_any(fields: List[str], values: List[str]) -> Dict[str, Any]:
    """Any of values across any fields."""
    return {"bool": {"should": [{"terms": {f: values}} for f in fields], "minimum_should_match": 1}}

def _dis_max_once(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Take the best scoring match only (prevents double counting)."""
    return {"dis_max": {"tie_breaker": 0.0, "queries": queries}}

import math
from typing import Any, Dict, List, Optional, Tuple, Union

def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]

def _get_by_dotted_path(obj: Dict[str, Any], path: str) -> Any:
    """
    Resolve dotted paths like 'Metadata.likes' from a nested _source dict.
    Returns None if missing.
    """
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

def normalize_es_id(es_id: str) -> str:
    out = es_id.replace("__", "/")
    if out.endswith(".json"):
        out = out[:-5]
    return out

def _normalize_value_key(value: Any) -> str:
    return str(value).strip().lower()


def _as_boolish_string(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return "true"
        if v in {"false", "0", "no", "n"}:
            return "false"
    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    boolish = _as_boolish_string(value)
    if boolish is not None:
        return boolish
    s = str(value).strip().lower()
    s = s.replace("__", "/")
    if s.endswith('.json'):
        s = s[:-5]
    s = re.sub(r'[_\-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _lexical_variants(value: Any) -> List[str]:
    if value is None:
        return []
    raw = str(value).strip()
    if not raw:
        return []

    variants = {
        raw,
        raw.lower(),
        raw.replace('-', ' '),
        raw.replace('-', '_'),
        raw.replace('_', '-'),
        raw.replace('_', ' '),
        raw.replace('/', ' / '),
    }

    norm = _normalize_text(raw)
    if norm:
        variants.add(norm)
        variants.add(norm.replace(' ', '-'))
        variants.add(norm.replace(' ', '_'))

    boolish = _as_boolish_string(value)
    if boolish is not None:
        variants.update({boolish, boolish.title(), boolish.upper()})

    return [v for v in dict.fromkeys(v.strip() for v in variants if str(v).strip())]


def _feature_value_variants(feature_key: str, value: Any) -> List[str]:
    variants = _lexical_variants(value)
    norm = _normalize_text(value)

    def add_prefixed(prefix: str):
        if norm:
            variants.append(f"{prefix}:{norm}")
            variants.append(f"{prefix}:{norm.replace(' ', '-')}")
            variants.append(f"{prefix}:{norm.replace(' ', '_')}")

    if feature_key == "license_name":
        add_prefixed("license")
    elif feature_key == "datasets":
        add_prefixed("dataset")
    elif feature_key == "objective":
        # Help phrases like pre-trained / pretrained / pre training line up.
        if "pre" in norm and "train" in norm:
            variants.extend(["pretrained", "pre trained", "pre-training", "pre training", "pretrained model"])
        if "embedding" in norm:
            variants.extend(["embedding", "embeddings", "text embedding", "text embeddings"])
        if "instruction" in norm:
            variants.extend(["instruction tuning", "instruction tuned", "sft", "supervised finetuning"])
        if "safety" in norm:
            variants.extend(["safety policy", "safe", "harmlessness"])
        if "adapt" in norm:
            variants.extend(["adaptation", "adaptable", "adapter", "adapters", "versatile", "versatility"])
        if "rl" in norm or "reinforcement" in norm:
            variants.extend(["ppo", "rlhf", "reward model", "human feedback", "human feedback learning"])
    elif feature_key in {"task", "task_alias"}:
        if "language" in norm and "model" in norm:
            variants.extend(["text-generation", "language model", "large language model", "llm", "causal lm"])
        if "text to text" in norm or "text-to-text" in norm or "text2text" in norm:
            variants.extend(["text2text-generation", "seq2seq", "sequence to sequence generation", "t5", "mt5"])
        if "translation" in norm:
            variants.extend(["translation", "machine translation", "multilingual machine translation", "many-to-many translation"])
    elif feature_key in {"domain", "domain_alias"}:
        if "medical" in norm:
            variants.extend(["clinical", "biomedical", "healthcare", "medicine"])
    elif feature_key == "language":
        if norm == "chinese":
            variants.extend(["zh", "zh-cn", "mandarin", "mandarin chinese"])
        elif norm == "arabic":
            variants.extend(["ar", "msa", "modern standard arabic"])
        elif norm == "multilingual":
            variants.extend(["many languages", "multi language", "cross language"])
    elif feature_key == "gated":
        boolish = _as_boolish_string(value)
        if boolish is not None:
            variants.extend([boolish, boolish.title(), boolish.upper()])

    return [v for v in dict.fromkeys(v.strip() for v in variants if str(v).strip())]


def _canonical_compare_variants(value: Any) -> List[str]:
    norm = _normalize_text(value)
    if not norm:
        return []
    variants = {norm}
    for prefix in ("license:", "dataset:"):
        if norm.startswith(prefix):
            variants.add(norm[len(prefix):].strip())
        else:
            variants.add(f"{prefix}{norm}")
    return list(variants)




def _tokenize_text(value: Any) -> List[str]:
    norm = _normalize_text(value)
    if not norm:
        return []
    return [tok for tok in re.split(r"\s+", norm) if tok]


def _flatten_field_text(source: Dict[str, Any], field: str) -> str:
    value = _get_by_dotted_path(source, field)
    vals = _as_list(value)
    parts: List[str] = []
    for v in vals:
        norm = _normalize_text(v)
        if norm:
            parts.append(norm)
    return " ".join(parts)


def _combined_fields_text(source: Dict[str, Any], fields: List[str]) -> str:
    return " ".join(part for part in (_flatten_field_text(source, f) for f in fields) if part).strip()


def _phrase_match_in_fields(source: Dict[str, Any], fields: List[str], query: Any) -> bool:
    q = _normalize_text(query)
    if not q:
        return False
    for field in fields:
        if q in _flatten_field_text(source, field):
            return True
    return q in _combined_fields_text(source, fields)


def _cross_field_token_match(source: Dict[str, Any], fields: List[str], query: Any, *, min_ratio: float = 0.6) -> bool:
    tokens = _tokenize_text(query)
    if not tokens:
        return False
    combined = _combined_fields_text(source, fields)
    if not combined:
        return False

    matched = 0
    for tok in tokens:
        if tok in combined:
            matched += 1
    if len(tokens) == 1:
        return matched == 1
    return (matched / max(1, len(tokens))) >= float(min_ratio)

def _quality_attr_to_mapping_key(attr_name: str) -> str:
    return f"{attr_name.lower()}_field"


def _quality_attr_to_default_path(attr_name: str) -> str:
    return f"Quality.{attr_name.replace('_', ' ')}.score"


# ----------------------------
# Core: Relaxation-driven ES Query Builder
# ----------------------------

class ESQueryBuilderAdaptive:
    """
    Should-only Elasticsearch query builder.

    Positive matching always uses `should` clauses from the beginning.
    For every user value, the builder emits:
      - canonical matches
      - grams / lexical variants
      - semantic synonyms
      - grams of semantic synonyms

    Match types are boosted by fidelity:
      - canonical: 1.00
      - grams: 0.95
      - synonyms: 0.80
      - grams_of_synonyms: 0.75

    There are no relaxation stages. Ranking remains additive and
    double-counting is prevented with global dis_max per user value.
    """

    MATCH_TYPE_BOOSTS: Dict[str, float] = {
        "canonical": 1.00,
        "grams": 0.95,
        "synonyms": 0.80,
        "grams_of_synonyms": 0.75,
    }

    def __init__(
        self,
        mapping: MappingType,
        *,
        base_signal_weights: Optional[Dict[str, float]] = None,
        priority_multipliers: Optional[Dict[PreferencePriority, float]] = None,
        feature_weight_groups: Optional[Dict[str, Dict[str, float]]] = None,
        boost_config: Optional[Dict[str, Dict[str, float]]] = None,
        size: int = 5,
        target_hits: int = 20,
        max_relax_steps: int = 50,
        synonym_min_conf: float = 0.20,
        max_synonyms_per_value: int = 10,
        tier_boost_start: float = 100.0,
        tier_boost_step: float = 100.0,
        rank_max_boost: float = 50.0,
        minimum_should_match: int = 1,
    ):
        self.mapping = mapping
        self.size = size
        self.target_hits = target_hits
        self.max_relax_steps = max_relax_steps

        self.resolver = AliasResolver(
            synonym_provider_factory=make_embedding_provider_factory(min_similarity=0.5),
            max_synonyms=max_synonyms_per_value,
        )

        default_feature_weight_groups: Dict[str, Dict[str, float]] = {
            "essential": {
                "task": 8.0,
                "domain": 2.0,
                "author": 3.0,
                "objective": 2.0,
            },
            "preference": {
                "license_name": 1.0,
                "library_name": 1.0,
                "basemodels": 1.0,
                "datasets": 1.0,
                "language": 1.0,
                "metrics": 1.0,
            },
            "functional": {
                "functional_item": 5.0,
            },
            "quality": {
                "Functional_Suitability": 1.5,
                "Compatibility": 1.0,
                "Performance_Efficiency": 1.0,
                "Reliability": 1.0,
                "Interaction_Capability": 1.0,
                "Security": 1.0,
                "Maintainability": 1.0,
                "Flexibility": 1.0,
            },
            "rank": {
                "likes": 0.8,
                "downloads_last_30_days": 1.2,
                "Functional_Suitability": 1.5,
                "Compatibility": 1.0,
                "Performance_Efficiency": 1.0,
                "Reliability": 1.0,
                "Interaction_Capability": 1.0,
                "Security": 1.0,
                "Maintainability": 1.0,
                "Flexibility": 1.0,
            },
        }

        if feature_weight_groups:
            for section, weights in feature_weight_groups.items():
                default_feature_weight_groups.setdefault(section, {}).update(weights)

        if base_signal_weights:
            compatibility_map = {
                "task_match": ("essential", "task"),
                "domain_match": ("essential", "domain"),
                "author_match": ("essential", "author"),
                "tag_match": ("preference", "license_name"),
                "functional_match": ("functional", "functional_item"),
                "quality_match": ("quality", "Functional_Suitability"),
            }
            for key, value in base_signal_weights.items():
                if key in compatibility_map:
                    section, name = compatibility_map[key]
                    default_feature_weight_groups.setdefault(section, {})[name] = float(value)
                else:
                    default_feature_weight_groups.setdefault("legacy", {})[key] = float(value)

        self.feature_weight_groups = default_feature_weight_groups
        self.base_signal_weights = {
            "tag_match": self.feature_weight_groups["preference"]["license_name"],
            "author_match": self.feature_weight_groups["essential"]["author"],
            "task_match": self.feature_weight_groups["essential"]["task"],
            "domain_match": self.feature_weight_groups["essential"]["domain"],
            "functional_match": self.feature_weight_groups["functional"]["functional_item"],
            "quality_match": self.feature_weight_groups["quality"]["Functional_Suitability"],
        }

        self.priority_multipliers = priority_multipliers or {
            "must": 1.6,
            "strong_prefer": 1.3,
            "prefer": 1.0,
            "avoid": 0.0,
        }

        default_boost_config: Dict[str, Dict[str, float]] = {
            "tier": {
                "start": float(tier_boost_start),
                "step": float(tier_boost_step),
            },
            "rank": {
                "max": float(rank_max_boost),
            },
            "match_mode": {
                "grams_factor": 0.99,
                "cross_fields_factor": 0.92,
                "phrase_factor": 1.05,
            },
        }
        if boost_config:
            for section, cfg in boost_config.items():
                default_boost_config.setdefault(section, {}).update(cfg)
        self.boost_config = default_boost_config

        self.synonym_min_conf = float(synonym_min_conf)
        self.max_synonyms_per_value = int(max_synonyms_per_value)

        self.tier_boost_start = float(self.boost_config["tier"]["start"])
        self.tier_boost_step = float(self.boost_config["tier"]["step"])
        self.rank_max_boost = float(self.boost_config["rank"]["max"])
        self.minimum_should_match = int(max(0, minimum_should_match))
        match_mode_cfg = self.boost_config.get("match_mode", {}) or {}
        self.grams_factor = float(match_mode_cfg.get("grams_factor") or 0.99)
        self.cross_fields_factor = float(match_mode_cfg.get("cross_fields_factor") or 0.92)
        self.phrase_factor = float(match_mode_cfg.get("phrase_factor") or 1.05)

        # ranking signals (optional “global” popularity/quality)
        self.rank_field_weights = dict(self.feature_weight_groups.get("rank", {}))
        self.rank_functions: List[Dict[str, Any]] = self._default_rank_functions()

    # ----------------------------
    # Mapping helpers
    # ----------------------------

    def _fields(self, field_key: str) -> List[str]:
        v = self.mapping.get(field_key)
        if v is None:
            raise KeyError(f"Missing mapping for field_key={field_key!r}")
        return [v] if isinstance(v, str) else list(v)

    def _first_field(self, field_key: str) -> str:
        return self._fields(field_key)[0]

    def _priority_weight(self, priority: PreferencePriority, base: float) -> float:
        return float(base) * float(self.priority_multipliers.get(priority, 1.0))

    def _value_key(self, value: Any) -> str:
        return _normalize_value_key(value)

    def _functional_candidate_field_keys(self) -> List[str]:
        configured = self.mapping.get("functional_search_fields") or self.mapping.get("functional_fields")
        if configured is not None:
            return list(configured) if isinstance(configured, list) else [configured]

        return [
            "task_field",
            "domain_field",
            "author_field",
            "objective_field",
            "license_field",
            "library_name_field",
            "basemodels_field",
            "datasets_field",
            "language_field",
            "metrics_field",
        ]

    def _functional_feature_fields(self) -> List[str]:
        fields: List[str] = []
        seen = set()
        for key in self._functional_candidate_field_keys():
            if key not in self.mapping:
                continue
            for field in self._fields(key):
                if field not in seen:
                    fields.append(field)
                    seen.add(field)
        return fields

    def _quality_feature_specs(self) -> List[Tuple[str, str, float]]:
        specs: List[Tuple[str, str, float]] = []
        for attr_name, weight in self.feature_weight_groups.get("quality", {}).items():
            mapping_key = _quality_attr_to_mapping_key(attr_name)
            default_path = _quality_attr_to_default_path(attr_name)
            field_path = self.mapping.get(mapping_key, default_path)
            specs.append((attr_name, field_path, float(weight)))
        return specs

    def _rank_signal_specs(self) -> List[Tuple[str, str, Optional[str], float, float]]:
        specs: List[Tuple[str, str, Optional[str], float, float]] = []

        likes_field = self.mapping.get("likes_field", "Metadata.likes")
        downloads_30d_field = self.mapping.get("downloads_30d_field", "Metadata.downloads_last_30_days")

        if self.rank_field_weights.get("likes", 0.0) > 0:
            specs.append(("likes", likes_field, "log1p", 0.0, float(self.rank_field_weights["likes"])))
        if self.rank_field_weights.get("downloads_last_30_days", 0.0) > 0:
            specs.append(("downloads_last_30_days", downloads_30d_field, "log1p", 0.0, float(self.rank_field_weights["downloads_last_30_days"])))

        for attr_name, _, _ in self._quality_feature_specs():
            weight = float(self.rank_field_weights.get(attr_name, 0.0))
            if weight <= 0:
                continue
            field_path = self.mapping.get(_quality_attr_to_mapping_key(attr_name), _quality_attr_to_default_path(attr_name))
            specs.append((attr_name, field_path, None, 0.0, weight))

        return specs

    # ----------------------------
    # GRAMS/SYNS BY VALUE
    # ----------------------------
    def _flat_grams(self, fg: FeatureGroup) -> List[str]:
        out: List[str] = []
        for v in fg.include:
            out.extend(fg.grams_by_value.get(v, []))
        # optional dedupe
        return list(dict.fromkeys(out))
    

    def _flat_syns(self, fg: FeatureGroup) -> List[str]:
        out: List[str] = []
        for v in fg.include:
            for s, _ in fg.syn_by_value.get(v, []):
                out.append(s)
        return list(dict.fromkeys(out))

    
    def _grams_of_synonyms_by_value(self, fg: FeatureGroup, user_value: Any) -> List[str]:
        syn_terms = [syn for syn, _ in fg.syn_by_value.get(user_value, [])]
        if not syn_terms:
            return []
        candidates_sources = [s.partition(".")[2] or s for s in fg.fields]
        try:
            grouped = self.resolver.resolve_grams_grouped(candidates_sources, syn_terms)
        except Exception:
            grouped = {}
        out: List[str] = []
        for syn in syn_terms:
            out.extend(grouped.get(syn, []))
            out.extend(v for v in _lexical_variants(syn) if _normalize_text(v) != _normalize_text(syn))
        return list(dict.fromkeys(v for v in out if str(v).strip()))

    def _add_value_candidates(
        self,
        *,
        value_key: str,
        value_queries: Dict[str, List[Dict[str, Any]]],
        fg: FeatureGroup,
        val: str,
        feat_weight: float,
    ):
        if feat_weight <= 0:
            return

        canonical_boost = feat_weight * float(self.MATCH_TYPE_BOOSTS["canonical"])
        value_queries[value_key].append(
            _wrap_constant_score(_terms_many(fg.fields, [val], k=1), canonical_boost)
        )
        for text_clause, factor in self._text_query_clauses_for_value(self._semantic_text_fields(fg), val):
            value_queries[value_key].append(
                _wrap_constant_score(text_clause, canonical_boost * float(factor))
            )

        grams_list = fg.grams_by_value.get(val, [])
        if grams_list:
            value_queries[value_key].append(
                _wrap_constant_score(
                    _terms_many(fg.fields, grams_list, k=1),
                    feat_weight * float(self.MATCH_TYPE_BOOSTS["grams"])
                )
            )

        syn_list = [syn for syn, _ in fg.syn_by_value.get(val, [])[: self.max_synonyms_per_value]]
        if syn_list:
            value_queries[value_key].append(
                _wrap_constant_score(
                    _terms_many(fg.fields, syn_list, k=1),
                    feat_weight * float(self.MATCH_TYPE_BOOSTS["synonyms"])
                )
            )

        syn_grams = self._grams_of_synonyms_by_value(fg, val)
        if syn_grams:
            value_queries[value_key].append(
                _wrap_constant_score(
                    _terms_many(fg.fields, syn_grams, k=1),
                    feat_weight * float(self.MATCH_TYPE_BOOSTS["grams_of_synonyms"])
                )
            )

    def _global_value_shoulds(self, groups: List[FeatureGroup]) -> List[Dict[str, Any]]:
        """
        Build ONE dis_max per user value across ALL features.
        Prevents cross-feature double counting for the same value.
        """
        value_queries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for fg in groups:
            if not fg.include or fg.priority == "avoid":
                continue

            feat_weight = self._priority_weight(fg.priority, fg.base_weight)
            if feat_weight <= 0:
                continue

            for val in fg.include:
                value_key = self._value_key(val)
                self._add_value_candidates(
                    value_key=value_key,
                    value_queries=value_queries,
                    fg=fg,
                    val=val,
                    feat_weight=float(feat_weight),
                )

        shoulds: List[Dict[str, Any]] = []
        for _, queries in value_queries.items():
            if queries:
                shoulds.append(_dis_max_once(queries))
        return shoulds

    def _best_match_for_value_detail(
        self,
        fg: FeatureGroup,
        source: Dict[str, Any],
        user_value: Any,
        feat_weight: float,
    ) -> Tuple[float, Dict[str, Any]]:
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        semantic_fields = self._semantic_text_fields(fg)

        if self._doc_has_term_in_any_field(source, fg.fields, user_value) or \
           _cross_field_token_match(source, semantic_fields, user_value) or \
           _phrase_match_in_fields(source, semantic_fields, user_value):
            score = float(feat_weight) * float(self.MATCH_TYPE_BOOSTS["canonical"])
            candidates.append((score, {
                "match_type": "canonical",
                "matched_term": user_value,
                "boost_factor": float(self.MATCH_TYPE_BOOSTS["canonical"]),
                "score": score,
            }))

        for gram in fg.grams_by_value.get(user_value, []):
            if self._doc_has_term_in_any_field(source, fg.fields, gram):
                score = float(feat_weight) * float(self.MATCH_TYPE_BOOSTS["grams"])
                candidates.append((score, {
                    "match_type": "grams",
                    "matched_term": gram,
                    "boost_factor": float(self.MATCH_TYPE_BOOSTS["grams"]),
                    "score": score,
                }))
                break

        for syn, _ in fg.syn_by_value.get(user_value, [])[: self.max_synonyms_per_value]:
            if self._doc_has_term_in_any_field(source, fg.fields, syn):
                score = float(feat_weight) * float(self.MATCH_TYPE_BOOSTS["synonyms"])
                candidates.append((score, {
                    "match_type": "synonyms",
                    "matched_term": syn,
                    "boost_factor": float(self.MATCH_TYPE_BOOSTS["synonyms"]),
                    "score": score,
                }))
                break

        for syn_gram in self._grams_of_synonyms_by_value(fg, user_value):
            if self._doc_has_term_in_any_field(source, fg.fields, syn_gram):
                score = float(feat_weight) * float(self.MATCH_TYPE_BOOSTS["grams_of_synonyms"])
                candidates.append((score, {
                    "match_type": "grams_of_synonyms",
                    "matched_term": syn_gram,
                    "boost_factor": float(self.MATCH_TYPE_BOOSTS["grams_of_synonyms"]),
                    "score": score,
                }))
                break

        if not candidates:
            return 0.0, {
                "match_type": "none",
                "matched_term": None,
                "boost_factor": 0.0,
                "score": 0.0,
            }

        return max(candidates, key=lambda x: x[0])

    def display_score_for_hit(
        self,
        groups: List[FeatureGroup],
        hit: Dict[str, Any],
        *,
        fixed_max_score: float,
    ) -> float:
        source = hit.get("_source", {}) or {}
        best_by_value: Dict[str, float] = defaultdict(float)

        for fg in groups:
            if not fg.include or fg.priority == "avoid":
                continue

            feat_weight = self._priority_weight(fg.priority, fg.base_weight)
            if feat_weight <= 0:
                continue

            for user_value in fg.include:
                value_key = self._value_key(user_value)
                score, _ = self._best_match_for_value_detail(fg, source, user_value, feat_weight)
                if score > best_by_value[value_key]:
                    best_by_value[value_key] = score

        feature_total = float(sum(best_by_value.values()))
        denom = max(1e-6, float(fixed_max_score))
        return float(min(100.0, (feature_total / denom) * 100.0))

    def _feature_total_global_by_value(
        self,
        groups: List[FeatureGroup],
        source: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Dict[str, Any]]]:
        best_score_by_value: Dict[str, float] = defaultdict(float)
        winners: Dict[str, Dict[str, Any]] = {}

        for fg in groups:
            if not fg.include or fg.priority == "avoid":
                continue

            feat_weight = self._priority_weight(fg.priority, fg.base_weight)
            if feat_weight <= 0:
                continue

            for user_value in fg.include:
                value_key = self._value_key(user_value)
                score, detail = self._best_match_for_value_detail(fg, source, user_value, feat_weight)
                if score > best_score_by_value[value_key]:
                    best_score_by_value[value_key] = score
                    winners[value_key] = {
                        "user_value": user_value,
                        "feature_key": fg.feature_key,
                        "priority": fg.priority,
                        "effective_feat_weight": feat_weight,
                        "score": float(score),
                        **detail,
                    }

        total = float(sum(best_score_by_value.values()))
        return total, winners
    # ----------------------------
    # Optional rank functions
    # ----------------------------

    def _default_rank_functions(self) -> List[Dict[str, Any]]:
        functions: List[Dict[str, Any]] = []
        for _, field_path, modifier, missing, weight in self._rank_signal_specs():
            body: Dict[str, Any] = {"field": field_path, "missing": float(missing)}
            if modifier:
                body["modifier"] = modifier
            functions.append({"field_value_factor": body, "weight": float(weight)})
        return functions

    def _soften_semantic_priority(self, feature_key: str, priority: PreferencePriority) -> PreferencePriority:
        if feature_key in {"objective", "domain", "domain_alias"} and priority == "must":
            return "strong_prefer"
        return priority

    def _safe_pref_priority(self, pref: Any, default: PreferencePriority = "prefer") -> PreferencePriority:
        raw = getattr(pref, "priority", default) or default
        return raw if raw in {"must", "strong_prefer", "prefer", "avoid"} else default

    def _semantic_text_fields(self, fg: FeatureGroup) -> List[str]:
        out: List[str] = []
        seen = set()
        for field in fg.fields:
            if field and field not in seen:
                out.append(field)
                seen.add(field)
        return out

    def _text_query_clauses_for_value(self, fields: List[str], value: Any) -> List[Tuple[Dict[str, Any], float]]:
        if not isinstance(value, str):
            return []
        query_text = str(value).strip()
        if not query_text:
            return []
        fields = [f for f in fields if f]
        if not fields:
            return []

        clauses: List[Tuple[Dict[str, Any], float]] = []
        clauses.append(({
            "multi_match": {
                "query": query_text,
                "fields": fields,
                "type": "cross_fields",
                "operator": "and",
            }
        }, self.cross_fields_factor))

        if len(_tokenize_text(query_text)) >= 2:
            clauses.append(({
                "multi_match": {
                    "query": query_text,
                    "fields": fields,
                    "type": "phrase",
                }
            }, self.phrase_factor))

        return clauses

    def _relaxation_priority_bucket(self, fg: FeatureGroup) -> Tuple[int, int]:
        semantic_first = {"objective", "domain", "task", "task_alias", "domain_alias"}
        structured_late = {"language", "license_name", "gated", "datasets", "author"}

        if fg.feature_key in semantic_first or fg.feature_key.startswith("functional_"):
            return (0, 0)
        if fg.feature_key in structured_late:
            return (2, 0)
        return (1, 0)

    # ----------------------------
    # Build FeatureGroups from your FeatureBundle
    # ----------------------------

    def build_feature_groups(
        self,
        features: Any,  # your FeatureBundle
    ) -> List[FeatureGroup]:
        groups: List[FeatureGroup] = []

        essential = getattr(features, "essential", None)

        task_pref = getattr(essential, "task", None)
        if task_pref and getattr(task_pref, "include", None):
            task_priority = self._safe_pref_priority(task_pref, "prefer")
            if task_priority in ("must", "strong_prefer"):
                task_priority = "prefer"
            fg = self._make_categorical_group(
                feature_key="task",
                field_key="task_field",
                pref=task_pref,
                base_weight=self.feature_weight_groups["essential"].get("task", 8.0),
                relaxable=True,
                force_priority=task_priority,
            )
            groups.append(fg)

        task_alias_pref = getattr(essential, "task_aliases", None)
        if task_alias_pref and getattr(task_alias_pref, "include", None):
            alias_priority = self._safe_pref_priority(task_alias_pref, "strong_prefer")
            if alias_priority == "must":
                alias_priority = "strong_prefer"
            groups.append(self._make_categorical_group(
                feature_key="task_alias",
                field_key="task_field",
                pref=task_alias_pref,
                base_weight=self.feature_weight_groups["essential"].get("task", 8.0) * 0.85,
                relaxable=True,
                force_priority=alias_priority,
            ))

        for key, field_key in [
            ("domain", "domain_field"),
            ("author", "author_field"),
            ("objective", "objective_field"),
        ]:
            pref = getattr(essential, key, None)
            if pref and getattr(pref, "include", None):
                groups.append(self._make_categorical_group(
                    feature_key=key,
                    field_key=field_key,
                    pref=pref,
                    base_weight=self.feature_weight_groups["essential"].get(key, 1.0),
                    relaxable=True,
                    force_priority=self._soften_semantic_priority(key, self._safe_pref_priority(pref)),
                ))

        domain_alias_pref = getattr(essential, "domain_aliases", None)
        if domain_alias_pref and getattr(domain_alias_pref, "include", None):
            alias_priority = self._safe_pref_priority(domain_alias_pref, "strong_prefer")
            if alias_priority == "must":
                alias_priority = "strong_prefer"
            groups.append(self._make_categorical_group(
                feature_key="domain_alias",
                field_key="domain_field",
                pref=domain_alias_pref,
                base_weight=self.feature_weight_groups["essential"].get("domain", 2.0) * 0.9,
                relaxable=True,
                force_priority=alias_priority,
            ))

        for key, field_key in [
            ("license_name", "license_field"),
            ("library_name", "library_name_field"),
            ("basemodels", "basemodels_field"),
            ("datasets", "datasets_field"),
            ("language", "language_field"),
            ("metrics", "metrics_field"),
            ("gated", "gated_field"),
        ]:
            pref = getattr(getattr(features, "preferences", None), key, None)
            if pref and (getattr(pref, "include", None) or getattr(pref, "exclude", None)):
                groups.append(self._make_categorical_group(
                    feature_key=key,
                    field_key=field_key,
                    pref=pref,
                    base_weight=self.feature_weight_groups["preference"].get(key, 1.0),
                    relaxable=True,
                ))

        functional = getattr(features, "functional", None)
        functional_items = list(getattr(functional, "F_features", None) or [])
        functional_fields = self._functional_feature_fields()
        if functional_items and functional_fields:
            for idx, item in enumerate(functional_items):
                groups.append(self._make_direct_group(
                    feature_key=f"functional_{idx}",
                    include=[item],
                    fields=functional_fields,
                    priority="prefer",
                    base_weight=self.feature_weight_groups["functional"].get("functional_item", 1.0),
                    relaxable=True,
                ))

        quality = getattr(features, "quality", None)
        if quality is not None:
            for attr_name, field_path, weight in self._quality_feature_specs():
                raw_value = getattr(quality, attr_name, None)
                if raw_value is None:
                    continue
                groups.append(self._make_direct_group(
                    feature_key=attr_name,
                    include=[raw_value],
                    fields=[field_path],
                    priority="prefer",
                    base_weight=weight,
                    relaxable=True,
                ))

        return groups

    
    def build_tier_filter(self, groups: List[FeatureGroup]) -> Dict[str, Any]:
        """
        Build a high-confidence should-only clause for optional tier boosting.
        """
        should: List[Dict[str, Any]] = []
        must_not: List[Dict[str, Any]] = []

        for fg in groups:
            if fg.exclude:
                ex_grams = self.resolver.resolve_grams(
                    [s.partition(".")[2] or s for s in fg.fields],
                    fg.exclude
                )
                if ex_grams:
                    must_not.append(_terms_any(fg.fields, ex_grams))

        for fg in groups:
            if not fg.include or fg.priority not in ("must", "strong_prefer"):
                continue
            feat_weight = self._priority_weight(fg.priority, fg.base_weight)
            for val in fg.include:
                should.append(_wrap_constant_score(_terms_many(fg.fields, [val], k=1), feat_weight))

        tier_q: Dict[str, Any] = {"bool": {}}
        if should:
            tier_q["bool"]["should"] = should
            tier_q["bool"]["minimum_should_match"] = 1
        if must_not:
            tier_q["bool"]["must_not"] = must_not
        return tier_q

    def _make_categorical_group(
        self,
        *,
        feature_key: str,
        field_key: str,
        pref: Any,
        base_weight: float,
        relaxable: bool,
        force_priority: Optional[PreferencePriority] = None,
    ) -> FeatureGroup:
        pr: PreferencePriority = force_priority or getattr(pref, "priority", "prefer")
        include = list(getattr(pref, "include", None) or [])
        exclude = list(getattr(pref, "exclude", None) or [])
        fields = self._fields(field_key)

        # Resolve grams and synonyms using your resolver contract.
        # Resolve grams and synonyms using grouped resolver outputs (per user value).
        candidates_sources = [s.partition(".")[2] or s for s in fields]

        grams_by_value: Dict[str, List[str]] = {}
        syn_by_value: Dict[str, List[Tuple[str, float]]] = {}

        if include:
            grams_by_value = self.resolver.resolve_grams_grouped(candidates_sources, include)
            syn_by_value = self.resolver.resolve_syns_grouped(candidates_sources, include)

            # Add lightweight lexical/canonical variants so schema quirks and tag prefixes still match.
            for v in include:
                base_variants = _feature_value_variants(feature_key, v)
                existing_grams = list(grams_by_value.get(v, []))
                existing_grams.extend(x for x in base_variants if _normalize_text(x) != _normalize_text(v))
                grams_by_value[v] = list(dict.fromkeys(existing_grams))

                existing_syns = list(syn_by_value.get(v, []))
                seen_syn_norms = {_normalize_text(s) for s, _ in existing_syns}
                for variant in base_variants:
                    norm_variant = _normalize_text(variant)
                    if norm_variant and norm_variant != _normalize_text(v) and norm_variant not in seen_syn_norms:
                        existing_syns.append((variant, 0.97))
                        seen_syn_norms.add(norm_variant)
                syn_by_value[v] = existing_syns

            # Apply synonym_min_conf filtering here (AliasResolver may not know your threshold)
            if self.synonym_min_conf > 0:
                filtered: Dict[str, List[Tuple[str, float]]] = {}
                for v in include:
                    pairs = syn_by_value.get(v, [])
                    filtered[v] = [(s, float(w)) for (s, w) in pairs if float(w) >= self.synonym_min_conf]
                syn_by_value = filtered



        return FeatureGroup(
            feature_key=feature_key,
            priority=pr,
            include=include,
            exclude=exclude,
            fields=fields,
            grams_by_value=grams_by_value,
            syn_by_value=syn_by_value,  # <--- changed
            level=0,
            relaxable=relaxable,
            base_weight=float(base_weight),
        )


    def _make_direct_group(
        self,
        *,
        feature_key: str,
        include: List[Any],
        fields: List[str],
        priority: PreferencePriority = "prefer",
        base_weight: float = 1.0,
        relaxable: bool = True,
        exclude: Optional[List[Any]] = None,
    ) -> FeatureGroup:
        return FeatureGroup(
            feature_key=feature_key,
            priority=priority,
            include=list(include or []),
            exclude=list(exclude or []),
            fields=list(fields),
            grams_by_value={},
            syn_by_value={},
            level=0,
            relaxable=relaxable,
            base_weight=float(base_weight),
        )

    # SCORE REASONING

    SCORE_BREAKDOWN_TOLERANCE = 1e-6

    def _collect_needed_source_paths(self, groups: List[FeatureGroup]) -> List[str]:
        paths: List[str] = []
        for fg in groups:
            paths.extend(fg.fields)

        for _, field_path, _, _, _ in self._rank_signal_specs():
            paths.append(field_path)

        seen = set()
        out = []
        for p in paths:
            if p and p not in seen:
                out.append(p)
                seen.add(p)
        return out
    
    def _get_docvalue(self, hit: Dict[str, Any], field_path: str) -> Any:
        fields = hit.get("fields") or {}
        v = fields.get(field_path)
        if isinstance(v, list):
            return v[0] if v else None
        return v



    def _rank_function_contribs_from_hit(self, hit: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
        breakdown: List[Dict[str, Any]] = []
        total = 0.0

        def fvf(signal_name: str, field_path: str, weight: float, modifier: Optional[str], missing: float) -> None:
            nonlocal total, breakdown

            raw = self._get_docvalue(hit, field_path)
            if raw is None:
                source = hit.get("_source", {}) or {}
                raw = _get_by_dotted_path(source, field_path)

            val = float(raw) if raw is not None else float(missing)

            if modifier == "log1p":
                modded = math.log1p(max(0.0, val))
            else:
                modded = val

            contrib = modded * float(weight)
            total += contrib

            breakdown.append({
                "type": "field_value_factor",
                "signal": signal_name,
                "field": field_path,
                "raw_value": raw,
                "missing_used": (raw is None),
                "modifier": modifier or "none",
                "after_modifier": modded,
                "weight": float(weight),
                "contribution": contrib,
            })

        for signal_name, field_path, modifier, missing, weight in self._rank_signal_specs():
            fvf(signal_name, field_path, weight=weight, modifier=modifier, missing=missing)

        return total, breakdown


    def _doc_has_term_in_any_field(self, source: Dict[str, Any], fields: List[str], term: str) -> bool:
        term_variants = set(_canonical_compare_variants(term))
        term_variants.update(_normalize_text(v) for v in _lexical_variants(term))
        term_variants = {v for v in term_variants if v}

        for f in fields:
            v = _get_by_dotted_path(source, f)
            vals = _as_list(v)
            for candidate in vals:
                candidate_variants = set(_canonical_compare_variants(candidate))
                candidate_variants.update(_normalize_text(v) for v in _lexical_variants(candidate))
                candidate_variants = {v for v in candidate_variants if v}
                if candidate_variants & term_variants:
                    return True
        return False

    def _match_mode_for_group(
        self,
        fg: FeatureGroup,
    ) -> Literal["best", "canonical_only", "grams_only", "synonyms_only"]:
        if fg.priority in ("must", "strong_prefer"):
            if fg.level == 2:
                return "grams_only"
            if fg.level == 3:
                return "synonyms_only"
            return "best"
        return "best"

    def _best_match_for_value(
        self,
        fg: FeatureGroup,
        source: Dict[str, Any],
        user_value: str,
        feat_weight: float,
        *,
        grams_factor: Optional[float] = None,
        mode: Literal["best", "canonical_only", "grams_only", "synonyms_only"] = "best",
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Recompute the dis_max best-of for ONE user value using the same mode the
        query builder uses for the feature at its current relaxation level.
        """
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        effective_grams_factor = self.grams_factor if grams_factor is None else float(grams_factor)
        semantic_fields = self._semantic_text_fields(fg)

        if mode in ("best", "canonical_only"):
            if self._doc_has_term_in_any_field(source, fg.fields, user_value):
                candidates.append((feat_weight, {
                    "match_type": "canonical",
                    "matched_term": user_value,
                    "factor": 1.0,
                    "score": feat_weight,
                }))

            if _cross_field_token_match(source, semantic_fields, user_value):
                s = feat_weight * float(self.cross_fields_factor)
                candidates.append((s, {
                    "match_type": "cross_fields",
                    "matched_term": user_value,
                    "factor": float(self.cross_fields_factor),
                    "score": s,
                }))

            if _phrase_match_in_fields(source, semantic_fields, user_value):
                s = feat_weight * float(self.phrase_factor)
                candidates.append((s, {
                    "match_type": "phrase",
                    "matched_term": user_value,
                    "factor": float(self.phrase_factor),
                    "score": s,
                }))

        if mode in ("best", "grams_only"):
            grams_list = fg.grams_by_value.get(user_value, [])
            for g in grams_list:
                if self._doc_has_term_in_any_field(source, fg.fields, g):
                    s = feat_weight * effective_grams_factor
                    candidates.append((s, {
                        "match_type": "grams",
                        "matched_term": g,
                        "factor": effective_grams_factor,
                        "score": s,
                    }))
                    break

        if mode in ("best", "synonyms_only"):
            best_syn: Optional[Tuple[str, float]] = None
            syn_list = fg.syn_by_value.get(user_value, [])
            for syn, sim in syn_list[: self.max_synonyms_per_value]:
                if self._doc_has_term_in_any_field(source, fg.fields, syn):
                    sim_f = float(sim)
                    if best_syn is None or sim_f > best_syn[1]:
                        best_syn = (syn, sim_f)

            if best_syn is not None:
                syn, sim_f = best_syn
                s = feat_weight * sim_f
                candidates.append((s, {
                    "match_type": "synonym",
                    "matched_term": syn,
                    "factor": sim_f,
                    "score": s,
                }))

        if not candidates:
            return 0.0, {
                "match_type": "none",
                "matched_term": None,
                "factor": 0.0,
                "score": 0.0,
            }

        best_score, best_detail = max(candidates, key=lambda x: x[0])
        return best_score, best_detail

    def score_breakdown_for_hit(
        self,
        groups: List[FeatureGroup],
        hit: Dict[str, Any],
        *,
        fixed_max_score: float,
    ) -> Dict[str, Any]:
        """
        Build a full score breakdown for a single ES hit.
        """
        source = hit.get("_source", {}) or {}
        es_score = float(hit.get("_score") or 0.0)

        feature_rows: List[Dict[str, Any]] = []

        # Global feature score mirrors ES dis_max-by-value behavior.
        feature_total_global, winners = self._feature_total_global_by_value(groups, source)

        for fg in groups:
            if not fg.include:
                continue
            if fg.priority == "avoid":
                continue

            pr_mult = float(self.priority_multipliers.get(fg.priority, 1.0))
            base = float(fg.base_weight)
            feat_weight = base * pr_mult
            per_value: List[Dict[str, Any]] = []
            feat_sum_local = 0.0
            feat_sum_global = 0.0

            for user_value in fg.include:
                value_key = self._value_key(user_value)
                best_score, best_detail = self._best_match_for_value(
                    fg,
                    source,
                    user_value,
                    feat_weight,
                )
                feat_sum_local += best_score

                winner = winners.get(value_key)
                winner_for_this_feature = bool(
                    winner
                    and winner.get("feature_key") == fg.feature_key
                    and self._value_key(winner.get("user_value")) == value_key
                )
                global_score = float(winner.get("score", 0.0)) if winner_for_this_feature else 0.0
                feat_sum_global += global_score

                per_value.append({
                    "user_value": user_value,
                    "best": best_detail,
                    "global_value_winner": winner_for_this_feature,
                    "global_value_contribution": global_score,
                })

            feature_rows.append({
                "feature_key": fg.feature_key,
                "priority": fg.priority,
                "multipliers": {
                    "base_weight": base,
                    "priority_multiplier": pr_mult,
                },
                "effective_feat_weight": feat_weight,
                "per_value_dismax": per_value,
                "feature_contribution_local": feat_sum_local,
                "feature_contribution_global": feat_sum_global,
                "feature_score_0_100_global": min(100.0, (feat_sum_global / max(1e-6, float(fixed_max_score))) * 100.0),
            })

        rank_total, rank_rows = self._rank_function_contribs_from_hit(hit)

        raw_est = feature_total_global + rank_total

        denom = max(1e-6, float(fixed_max_score))
        feature_score_0_100 = min(100.0, (feature_total_global / denom) * 100.0)

        return {
            "es_score": es_score,
            "estimated_raw_score": raw_est,
            "feature_score_0_100": feature_score_0_100,
            "feature_total_est": feature_total_global,
            "value_winners": winners,
            "rank_total_est": rank_total,
            "features": feature_rows,
            "rank_functions": rank_rows,
            "notes": {
                "estimation": (
                    "Computed from _source values using the same multipliers/penalties "
                    "as query builder. feature_contribution_global mirrors the actual "
                    "global dis_max-by-value feature score used for normalization. "
                    "ES _score may differ if docvalues/_source differ, analyzers differ, "
                    "or if ES scoring differs from our assumptions."
                ),
            },
        }
    # ----------------------------
    # Build ES query from current FeatureGroup states
    # ----------------------------

    
    def build_query(self, groups: List[FeatureGroup], *, include_explain: bool = False,
                    tier_filters: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Returns an ES query dict using should-only positive matching.
        """
        should: List[Dict[str, Any]] = []
        must_not: List[Dict[str, Any]] = []

        for fg in groups:
            if fg.exclude:
                ex_grams = self.resolver.resolve_grams(
                    [s.partition(".")[2] or s for s in fg.fields],
                    fg.exclude
                )
                if ex_grams:
                    must_not.append(_terms_any(fg.fields, ex_grams))

        for fg in groups:
            if not fg.include:
                continue
            if fg.priority == "avoid":
                flat = self._flat_grams(fg)
                if flat:
                    must_not.append(_terms_any(fg.fields, flat))

        should = self._global_value_shoulds(groups)

        bool_query: Dict[str, Any] = {"bool": {}}
        if should:
            bool_query["bool"]["should"] = should
            bool_query["bool"]["minimum_should_match"] = int(max(0, self.minimum_should_match))
        else:
            bool_query["bool"]["minimum_should_match"] = 0
        if must_not:
            bool_query["bool"]["must_not"] = must_not

        rank_query: Dict[str, Any] = {
            "function_score": {
                "query": bool_query,
                "score_mode": "sum",
                "boost_mode": "sum",
                "functions": self.rank_functions,
                "max_boost": self.rank_max_boost,
            }
        }

        tier_filters = tier_filters or []
        if tier_filters:
            tier_funcs: List[Dict[str, Any]] = []
            for i, tf in enumerate(tier_filters):
                w = self.tier_boost_start - (i * self.tier_boost_step)
                if w <= 0:
                    break
                tier_funcs.append({"filter": tf, "weight": float(w)})
            base_query: Dict[str, Any] = {
                "function_score": {
                    "query": rank_query,
                    "score_mode": "sum",
                    "boost_mode": "sum",
                    "functions": tier_funcs,
                }
            }
        else:
            base_query = rank_query

        docvalue_fields = [field_path for _, field_path, _, _, _ in self._rank_signal_specs()]

        return {
            "explain": bool(include_explain),
            "size": self.size,
            "_source": {"includes": self._collect_needed_source_paths(groups)},
            "docvalue_fields": docvalue_fields,
            "query": base_query,
            "sort": [{"_score": {"order": "desc"}}],
        }
    # ----------------------------
    # Max-score computation (normalized scoring)
    # ----------------------------

    
    def compute_max_score(self, groups: List[FeatureGroup]) -> float:
        best_by_value: Dict[str, float] = {}
        for fg in groups:
            if not fg.include or fg.priority == "avoid":
                continue
            base = self._priority_weight(fg.priority, fg.base_weight) * float(self.MATCH_TYPE_BOOSTS["canonical"])
            for val in fg.include:
                key = self._value_key(val)
                if key not in best_by_value or base > best_by_value[key]:
                    best_by_value[key] = float(base)
        return float(sum(best_by_value.values()))


    # ----------------------------
    # Feasibility tests (relax only what cannot match)
    # ----------------------------

    
    def _best_match_for_value_mode(
        self,
        fg: FeatureGroup,
        source: Dict[str, Any],
        user_value: str,
        feat_weight: float,
        mode: Literal["best", "canonical_only", "grams_only", "synonyms_only"],
        *,
        grams_factor: Optional[float] = None,
    ) -> float:
        score, _ = self._best_match_for_value_detail(fg, source, user_value, feat_weight)
        return float(score)

    def _base_constraints_query(self, groups: List[FeatureGroup]) -> Dict[str, Any]:
        q: Dict[str, Any] = {"bool": {}}
        must_not: List[Dict[str, Any]] = []
        for fg in groups:
            if fg.exclude:
                ex_grams = self.resolver.resolve_grams(
                    [s.partition(".")[2] or s for s in fg.fields],
                    fg.exclude
                )
                if ex_grams:
                    must_not.append(_terms_any(fg.fields, ex_grams))
        if must_not:
            q["bool"]["must_not"] = must_not
        q["bool"]["minimum_should_match"] = 0
        return q

    
    def compare_fundle_to_sample(
        self,
        features: Any,
        sample_file: Union[str, os.PathLike, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare a feature bundle against a sample document and return the
        matched items with weights and match labels.
        """
        groups = self.build_feature_groups(features)

        if isinstance(sample_file, dict):
            sample_doc = sample_file
            sample_path = None
        else:
            sample_path = os.fspath(sample_file)
            with open(sample_path, "r") as f:
                sample_doc = json.load(f)

        per_feature: List[Dict[str, Any]] = []
        best_by_value: Dict[str, Dict[str, Any]] = {}
        total_score = 0.0
        max_score = max(1e-6, self.compute_max_score(groups))

        for fg in groups:
            if not fg.include or fg.priority == "avoid":
                continue

            feat_weight = self._priority_weight(fg.priority, fg.base_weight)
            feature_matches: List[Dict[str, Any]] = []

            for user_value in fg.include:
                score, detail = self._best_match_for_value_detail(fg, sample_doc, user_value, feat_weight)
                item = {
                    "feature_key": fg.feature_key,
                    "user_value": user_value,
                    "priority": fg.priority,
                    "base_weight": float(fg.base_weight),
                    "effective_weight": float(feat_weight),
                    "matched": score > 0.0,
                    "match_type": detail.get("match_type"),
                    "matched_term": detail.get("matched_term"),
                    "boost_factor": detail.get("boost_factor", 0.0),
                    "score": float(score),
                    "fields": list(fg.fields),
                }
                feature_matches.append(item)

                value_key = self._value_key(user_value)
                cur = best_by_value.get(value_key)
                if cur is None or float(item["score"]) > float(cur["score"]):
                    best_by_value[value_key] = item

            per_feature.append({
                "feature_key": fg.feature_key,
                "priority": fg.priority,
                "effective_weight": float(feat_weight),
                "matches": feature_matches,
            })

        matched_items = []
        for _, item in best_by_value.items():
            matched_items.append(item)
            total_score += float(item["score"])

        return {
            "sample_model_id": sample_doc.get("modelID") or sample_doc.get("model_id"),
            "sample_path": sample_path,
            "matched_items": matched_items,
            "per_feature": per_feature,
            "total_match_score": total_score,
            "normalized_match_score": min(100.0, (total_score / max_score) * 100.0),
            "match_type_boosts": dict(self.MATCH_TYPE_BOOSTS),
            "minimum_should_match": int(self.minimum_should_match),
        }
    
    def _ensure_dict(self, resp):
        # Elasticsearch 8.x returns ObjectApiResponse
        if hasattr(resp, "body"):
            return resp.body
        return resp
    
    def search(self, es_client: Any, index: str, features: Any, *, include_score_breakdown: bool = False, include_explain: bool = False, ):
        groups = self.build_feature_groups(features)
        fixed_max_score = max(1e-6, float(self.compute_max_score(groups)))

        q = self.build_query(groups, include_explain=include_explain, tier_filters=None)
        resp = es_client.search(index=index, body=q)
        resp_dict = self._ensure_dict(resp)

        hits_list = resp_dict.get("hits", {}).get("hits", [])

        for i, h in enumerate(hits_list):
            raw_id = h.get("_id")
            pretty_id = None
            if raw_id:
                pretty_id = raw_id.replace("__", "/")
                if pretty_id.endswith(".json"):
                    pretty_id = pretty_id[:-5]

            display_score = round(self.display_score_for_hit(groups, h, fixed_max_score=fixed_max_score), 2)
            new_hit = {}
            if "_id" in h:
                new_hit["_id"] = h["_id"]
            if pretty_id is not None:
                new_hit["pretty_id"] = pretty_id
            if "_score" in h:
                new_hit["_score"] = h["_score"]
            new_hit["display_score"] = display_score
            for k, v in h.items():
                if k in ("_id", "_score", "_source"):
                    continue
                new_hit[k] = v
            if "_source" in h:
                new_hit["_source"] = h["_source"]
            hits_list[i] = new_hit

        if include_score_breakdown:
            for h in hits_list:
                h["_score_breakdown"] = self.score_breakdown_for_hit(
                    groups,
                    h,
                    fixed_max_score=fixed_max_score,
                )
        else:
            for h in hits_list:
                h.pop("_score_breakdown", None)

        if not include_explain:
            for h in hits_list:
                h.pop("_explanation", None)

        return resp_dict, q, groups
