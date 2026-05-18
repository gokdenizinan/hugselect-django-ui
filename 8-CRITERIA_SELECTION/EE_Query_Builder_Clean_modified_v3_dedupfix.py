from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal, Union, MutableMapping

from datetime import datetime
from collections import defaultdict
import re
import os
import json
import hashlib
from threading import RLock


# ----------------------------
# Synonym / grams cache helpers
# ----------------------------

class SimpleSynonymCache:
    """
    Thread-safe cache that can be in-memory only or persisted to a JSON file.
    Persists in batches instead of flushing on every write.
    """

    def __init__(self, cache_path: Optional[str] = None, flush_every: int = 100):
        self.cache_path = cache_path
        self.flush_every = max(1, int(flush_every))
        self._lock = RLock()
        self._data: Dict[str, Any] = {}
        self._pending_writes = 0

        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self._data = loaded
            except Exception:
                self._data = {}

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            old_value = self._data.get(key, None)
            if old_value == value:
                return

            self._data[key] = value
            self._pending_writes += 1

            if self.cache_path and self._pending_writes >= self.flush_every:
                self._flush_unlocked()
                self._pending_writes = 0

    def flush(self) -> None:
        with self._lock:
            if self.cache_path and self._pending_writes > 0:
                self._flush_unlocked()
                self._pending_writes = 0

    def _flush_unlocked(self) -> None:
        if not self.cache_path:
            return
        tmp_path = f"{self.cache_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)
        os.replace(tmp_path, self.cache_path)

def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _cache_key(prefix: str, payload: Dict[str, Any]) -> str:
    digest = hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"

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
MappingValue = Union[str, List[str]]
MappingType = Dict[str, MappingValue]

# External contracts (yours)
# from EE_Alias_Creation import AliasResolver, make_embedding_provider_factory
from EE_Alias_Creation_cached import AliasResolver, make_embedding_provider_factory


# ----------------------------
# FeatureGroup: feature-by-feature matching metadata
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

    # scoring
    base_weight: float = 1.0                 # feature-level base weight
    per_value_weight: Optional[float] = None # if you want per-value weights


# ----------------------------
# Helpers: ES query building primitives
# ----------------------------

def _emit_range(field: str, body: Dict[str, Any]) -> Dict[str, Any]:
    return {"range": {field: body}}

def _wrap_constant_score(filter_clause: Dict[str, Any], boost: float) -> Dict[str, Any]:
    return {"constant_score": {"filter": filter_clause, "boost": float(boost)}}

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


def _dis_max_once(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Take the best scoring match only (prevents double counting)."""
    return {"dis_max": {"tie_breaker": 0.0, "queries": queries}}


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


def _canonical_token_signature(value: Any) -> str:
    norm = _normalize_text(value)
    if not norm:
        return ""
    tokens = re.findall(r"[a-z0-9]+", norm.casefold())
    return " ".join(tokens)


def _normalize_value_key(value: Any) -> str:
    """
    Canonical scoring key used to collapse near-identical user inputs into the
    same scoring bucket across feature groups.

    Examples:
      - image-classification == image classification == image_classification
      - license:mit == mit
      - dataset:glue == glue
    """
    candidates = []
    seen = set()
    for variant in _canonical_compare_variants(value) or [_normalize_text(value)]:
        key = _canonical_token_signature(variant)
        if key and key not in seen:
            candidates.append(key)
            seen.add(key)
    if candidates:
        return min(candidates, key=lambda s: (len(s), s))
    return _canonical_token_signature(value)


def _dedupe_values_preserve_order(values: List[Any]) -> List[Any]:
    out: List[Any] = []
    seen: set[str] = set()
    for value in values:
        key = _normalize_value_key(value)
        fallback = str(value).strip().lower()
        dedupe_key = key or fallback
        if not dedupe_key or dedupe_key in seen:
            continue
        out.append(value)
        seen.add(dedupe_key)
    return out


def _dedupe_scored_terms_preserve_best(
    values: List[Tuple[str, float]],
) -> List[Tuple[str, float]]:
    best_by_key: Dict[str, Tuple[str, float]] = {}
    order: List[str] = []
    for term, weight in values:
        key = _normalize_value_key(term) or _canonical_token_signature(term) or str(term).strip().lower()
        if not key:
            continue
        cur = best_by_key.get(key)
        if cur is None:
            best_by_key[key] = (term, float(weight))
            order.append(key)
            continue
        cur_term, cur_weight = cur
        new_weight = float(weight)
        if new_weight > cur_weight or (new_weight == cur_weight and len(str(term)) < len(str(cur_term))):
            best_by_key[key] = (term, new_weight)
    return [best_by_key[key] for key in order]


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


def _text_variants(value: Any) -> List[str]:
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
        raw.replace('/', ' '),
        raw.replace('/', '-'),
        raw.replace('/', '_'),
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


def _flatten_field_text(source: Dict[str, Any], field: str) -> str:
    value = _get_by_dotted_path(source, field)
    vals = _as_list(value)
    parts: List[str] = []
    for v in vals:
        norm = _normalize_text(v)
        if norm:
            parts.append(norm)
    return " ".join(parts)


def _quality_attr_to_mapping_key(attr_name: str) -> str:
    return f"{attr_name.lower()}_field"


def _quality_attr_to_default_path(attr_name: str) -> str:
    return f"Quality.{attr_name.replace('_', ' ')}.score"

def _coerce_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"(19|20)\d{2}", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _extract_year_from_datetime_like(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value if 0 <= value <= 9999 else None
    if isinstance(value, float):
        iv = int(value)
        return iv if 0 <= iv <= 9999 else None
    s = str(value).strip()
    if not s:
        return None

    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).year
    except Exception:
        pass

    m = re.search(r"(19|20)\d{2}", s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


# ----------------------------
# Core: Relaxation-driven ES Query Builder
# ----------------------------

class ESQueryBuilderAdaptive:
    """
    Should-only Elasticsearch query builder.

    Positive matching always uses `should` clauses from the beginning.
    For every user value, the builder emits:
      - canonical matches
      - grams / normalized variants
      - semantic synonyms

    Match types are boosted by fidelity:
      - canonical: 1.00
      - grams: configurable via grams_factor
      - synonyms: 0.80 × synonym confidence

    Ranking remains additive and
    double-counting is prevented with global dis_max per user value.
    """

    MATCH_TYPE_BOOSTS: Dict[str, float] = {
        "canonical": 1.00,
        "synonyms": 0.80,
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
        synonym_min_conf: float = 0.20,
        max_synonyms_per_value: int = 10,
        rank_max_boost: float = 50.0,
        minimum_should_match: int = 1,
        enable_rank_functions: bool = True,
        enable_quality_dimensions: bool = True,
        enable_feature_locations: bool = True,
        synonym_cache: Optional[MutableMapping[str, Any]] = None,
        synonym_cache_path: Optional[str] = None,
        use_synonym_cache: bool = True,
    ):
        self.mapping = mapping
        self.size = size
        self.target_hits = target_hits

        self.resolver = AliasResolver(
            synonym_provider_factory=make_embedding_provider_factory(min_similarity=0.5),
            max_synonyms=max_synonyms_per_value,
        )

        self.use_synonym_cache = bool(use_synonym_cache)
        if synonym_cache is not None:
            self.synonym_cache = synonym_cache
        elif synonym_cache_path:
            self.synonym_cache = SimpleSynonymCache(synonym_cache_path, flush_every=100)
        else:
            self.synonym_cache = SimpleSynonymCache()
        
        

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
            "rank": {
                "max": float(rank_max_boost),
            },
            "match_mode": {
                "grams_factor": 0.99,
            },
        }
        if boost_config:
            for section, cfg in boost_config.items():
                default_boost_config.setdefault(section, {}).update(cfg)
        self.boost_config = default_boost_config

        self.synonym_min_conf = float(synonym_min_conf)
        self.max_synonyms_per_value = int(max_synonyms_per_value)

        self.rank_max_boost = float(self.boost_config["rank"]["max"])
        self.minimum_should_match = int(max(0, minimum_should_match))
        self.enable_rank_functions = bool(enable_rank_functions)
        self.enable_quality_dimensions = bool(enable_quality_dimensions)
        self.enable_feature_locations = bool(enable_feature_locations)
        match_mode_cfg = self.boost_config.get("match_mode", {}) or {}
        self.grams_factor = float(match_mode_cfg.get("grams_factor") or 0.99)

        self.quality_feature_specs = self._build_quality_feature_specs()
        self.quality_feature_key_set = {attr_name for attr_name, _, _ in self.quality_feature_specs}

        # ranking signals (optional “global” popularity/quality)
        self.rank_field_weights = dict(self.feature_weight_groups.get("rank", {}))
        self.rank_signal_specs = self._build_rank_signal_specs()
        self.rank_functions: List[Dict[str, Any]] = self._default_rank_functions()

    # ----------------------------
    # Mapping helpers
    # ----------------------------

    def _filter_fields_by_toggles(self, fields: List[str]) -> List[str]:
        filtered: List[str] = []
        for field in fields:
            if not self.enable_feature_locations and field == "Features":
                continue
            filtered.append(field)
        return filtered

    def _fields(self, field_key: str) -> List[str]:
        v = self.mapping.get(field_key)
        if v is None:
            raise KeyError(f"Missing mapping for field_key={field_key!r}")
        fields = [v] if isinstance(v, str) else list(v)
        return self._filter_fields_by_toggles(fields)


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
        for item in self._functional_candidate_field_keys():
            if item in self.mapping:
                expanded = self._fields(item)
            else:
                expanded = self._filter_fields_by_toggles([item])
            for field in expanded:
                if field and field not in seen:
                    fields.append(field)
                    seen.add(field)
        return fields

    def _build_quality_feature_specs(self) -> List[Tuple[str, str, float]]:
        if not self.enable_quality_dimensions:
            return []
        specs: List[Tuple[str, str, float]] = []
        for attr_name, weight in self.feature_weight_groups.get("quality", {}).items():
            mapping_key = _quality_attr_to_mapping_key(attr_name)
            default_path = _quality_attr_to_default_path(attr_name)
            field_path = self.mapping.get(mapping_key, default_path)
            specs.append((attr_name, field_path, float(weight)))
        return specs

    def _quality_feature_keys(self) -> set[str]:
        return set(self.quality_feature_key_set)

    def _is_quality_feature(self, feature_key: str) -> bool:
        return feature_key in self.quality_feature_key_set

    def _quality_threshold_filter(self, fg: FeatureGroup) -> Optional[Dict[str, Any]]:
        if not self._is_quality_feature(fg.feature_key) or not fg.fields:
            return None
        return _emit_range(fg.fields[0], {"gt": 0.6})

    def _quality_boost_detail(self, fg: FeatureGroup, source: Dict[str, Any], feat_weight: float) -> Tuple[float, Dict[str, Any]]:
        if not self._is_quality_feature(fg.feature_key) or not fg.fields:
            return 0.0, {
                "match_type": "none",
                "matched_term": None,
                "boost_factor": 0.0,
                "score": 0.0,
            }
        raw_value = _get_by_dotted_path(source, fg.fields[0])
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            numeric_value = None
        if numeric_value is not None and numeric_value > 0.6:
            score = float(feat_weight)
            return score, {
                "match_type": "quality_threshold",
                "matched_term": numeric_value,
                "boost_factor": 1.0,
                "score": score,
            }
        return 0.0, {
            "match_type": "quality_below_threshold",
            "matched_term": raw_value,
            "boost_factor": 0.0,
            "score": 0.0,
        }

    def _build_rank_signal_specs(self) -> List[Tuple[str, str, Optional[str], float, float]]:
        if not self.enable_rank_functions:
            return []

        specs: List[Tuple[str, str, Optional[str], float, float]] = []

        likes_field = self.mapping.get("likes_field", "Metadata.likes")
        downloads_30d_field = self.mapping.get("downloads_30d_field", "Metadata.downloads_last_30_days")

        if self.rank_field_weights.get("likes", 0.0) > 0:
            specs.append(("likes", likes_field, "log1p", 0.0, float(self.rank_field_weights["likes"])))
        if self.rank_field_weights.get("downloads_last_30_days", 0.0) > 0:
            specs.append(("downloads_last_30_days", downloads_30d_field, "log1p", 0.0, float(self.rank_field_weights["downloads_last_30_days"])))

        if self.enable_quality_dimensions:
            for attr_name, _, _ in self.quality_feature_specs:
                weight = float(self.rank_field_weights.get(attr_name, 0.0))
                if weight <= 0:
                    continue
                field_path = self.mapping.get(_quality_attr_to_mapping_key(attr_name), _quality_attr_to_default_path(attr_name))
                specs.append((attr_name, field_path, None, 0.0, weight))

        return specs


    def _syn_cache_payload(
        self,
        *,
        op: str,
        candidates_sources: List[str],
        values: List[Any],
        feature_key: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "op": op,
            "candidates_sources": sorted(str(s) for s in candidates_sources),
            "values": [_normalize_text(v) for v in values],
            "feature_key": feature_key or "",
            "synonym_min_conf": float(self.synonym_min_conf),
            "max_synonyms_per_value": int(self.max_synonyms_per_value),
        }
        if extra:
            payload.update(extra)
        return payload

    def _cache_get_or_compute(self, cache_key: str, compute_fn):
        if not self.use_synonym_cache:
            return compute_fn()
        cached = self.synonym_cache.get(cache_key)
        if cached is not None:
            return cached
        value = compute_fn()
        try:
            self.synonym_cache[cache_key] = value
        except Exception:
            pass
        return value

    def _resolve_grams_grouped_cached(
        self,
        candidates_sources: List[str],
        include: List[Any],
        *,
        feature_key: str,
    ) -> Dict[str, List[str]]:
        if not include:
            return {}
        payload = self._syn_cache_payload(
            op="resolve_grams_grouped",
            candidates_sources=candidates_sources,
            values=include,
            feature_key=feature_key,
        )
        key = _cache_key("grams_grouped", payload)
        return self._cache_get_or_compute(
            key,
            lambda: self.resolver.resolve_grams_grouped(candidates_sources, include),
        )

    def _resolve_syns_grouped_cached(
        self,
        candidates_sources: List[str],
        include: List[Any],
        *,
        feature_key: str,
    ) -> Dict[str, List[Tuple[str, float]]]:
        if not include:
            return {}
        payload = self._syn_cache_payload(
            op="resolve_syns_grouped",
            candidates_sources=candidates_sources,
            values=include,
            feature_key=feature_key,
        )
        key = _cache_key("syns_grouped", payload)
        return self._cache_get_or_compute(
            key,
            lambda: self.resolver.resolve_syns_grouped(candidates_sources, include),
        )

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

        if self._is_quality_feature(fg.feature_key):
            quality_filter = self._quality_threshold_filter(fg)
            if quality_filter is not None:
                value_queries[value_key].append(
                    _wrap_constant_score(quality_filter, feat_weight * float(self.MATCH_TYPE_BOOSTS["canonical"]))
                )
            return

        canonical_boost = feat_weight * float(self.MATCH_TYPE_BOOSTS["canonical"])
        value_queries[value_key].append(
            _wrap_constant_score(_terms_many(fg.fields, [val], k=1), canonical_boost)
        )

        grams_list = fg.grams_by_value.get(val, [])
        if grams_list:
            value_queries[value_key].append(
                _wrap_constant_score(
                    _terms_many(fg.fields, grams_list, k=1),
                    feat_weight * float(self.grams_factor)
                )
            )

        syn_list = fg.syn_by_value.get(val, [])[: self.max_synonyms_per_value]
        for syn, confidence in syn_list:
            synonym_boost = feat_weight * float(self.MATCH_TYPE_BOOSTS["synonyms"]) * float(confidence)
            if synonym_boost <= 0:
                continue
            value_queries[value_key].append(
                _wrap_constant_score(
                    _terms_many(fg.fields, [syn], k=1),
                    synonym_boost
                )
            )

    def _global_value_shoulds(self, groups: List[FeatureGroup]) -> List[Dict[str, Any]]:
        """
        Build ONE dis_max per user value across ALL features.
        Prevents cross-feature double counting for the same value.
        """
        value_queries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for fg in groups:
            if not fg.include or fg.priority == "avoid" or fg.feature_key == "last_modified_year":
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
        if fg.feature_key == "last_modified_year":
            if self._doc_satisfies_last_modified(source, fg):
                doc_value = _get_by_dotted_path(source, fg.fields[0]) if fg.fields else None
                return 0.0, {
                    "match_type": "filter_pass",
                    "matched_term": doc_value,
                    "boost_factor": 0.0,
                    "score": 0.0,
                }
            doc_value = _get_by_dotted_path(source, fg.fields[0]) if fg.fields else None
            return 0.0, {
                "match_type": "filter_fail",
                "matched_term": doc_value,
                "boost_factor": 0.0,
                "score": 0.0,
            }

        if self._is_quality_feature(fg.feature_key):
            return self._quality_boost_detail(fg, source, feat_weight)

        candidates: List[Tuple[float, Dict[str, Any]]] = []

        if self._doc_has_term_in_any_field(source, fg.fields, user_value):
            score = float(feat_weight) * float(self.MATCH_TYPE_BOOSTS["canonical"])
            candidates.append((score, {
                "match_type": "canonical",
                "matched_term": user_value,
                "boost_factor": float(self.MATCH_TYPE_BOOSTS["canonical"]),
                "score": score,
            }))

        for gram in fg.grams_by_value.get(user_value, []):
            if self._doc_has_term_in_any_field(source, fg.fields, gram):
                score = float(feat_weight) * float(self.grams_factor)
                candidates.append((score, {
                    "match_type": "grams",
                    "matched_term": gram,
                    "boost_factor": float(self.grams_factor),
                    "score": score,
                }))
                break

        for syn, confidence in fg.syn_by_value.get(user_value, [])[: self.max_synonyms_per_value]:
            if self._doc_has_term_in_any_field(source, fg.fields, syn):
                applied_confidence = float(confidence)
                score = float(feat_weight) * float(self.MATCH_TYPE_BOOSTS["synonyms"]) * applied_confidence
                candidates.append((score, {
                    "match_type": "synonyms",
                    "matched_term": syn,
                    "boost_factor": float(self.MATCH_TYPE_BOOSTS["synonyms"]) * applied_confidence,
                    "synonym_confidence": applied_confidence,
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
            if fg.feature_key == "last_modified_year":
                if not self._doc_satisfies_last_modified(source, fg):
                    return 0.0
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


    # ----------------------------
    # Optional rank functions
    # ----------------------------

    def _default_rank_functions(self) -> List[Dict[str, Any]]:
        functions: List[Dict[str, Any]] = []
        for _, field_path, modifier, missing, weight in self.rank_signal_specs:
            body: Dict[str, Any] = {"field": field_path, "missing": float(missing)}
            if modifier:
                body["modifier"] = modifier
            functions.append({"field_value_factor": body, "weight": float(weight)})
        return functions

    def _safe_pref_priority(self, pref: Any, default: PreferencePriority = "prefer") -> PreferencePriority:
        raw = getattr(pref, "priority", default) or default
        return raw if raw in {"must", "strong_prefer", "prefer", "avoid"} else default


    def _extract_last_modified_pref(self, features: Any) -> Optional[Any]:
        preferences = getattr(features, "preferences", None)
        if preferences is None:
            return None
        for attr_name in ("last_modified_year", "last_modified", "max_last_modified_year"):
            pref = getattr(preferences, attr_name, None)
            if pref is not None:
                return pref
        return None

    def _last_modified_cutoff_year(self, fg: FeatureGroup) -> Optional[int]:
        if fg.feature_key != "last_modified_year" or not fg.include:
            return None
        return _coerce_year(fg.include[0])

    def _last_modified_filter_clause(self, fg: FeatureGroup) -> Optional[Dict[str, Any]]:
        cutoff_year = self._last_modified_cutoff_year(fg)
        if cutoff_year is None or not fg.fields:
            return None
        field = fg.fields[0]
        return _emit_range(field, {"lte": f"{cutoff_year}-12-31T23:59:59+00:00"})

    def _doc_satisfies_last_modified(self, source: Dict[str, Any], fg: FeatureGroup) -> bool:
        cutoff_year = self._last_modified_cutoff_year(fg)
        if cutoff_year is None or not fg.fields:
            return True
        doc_year = _extract_year_from_datetime_like(_get_by_dotted_path(source, fg.fields[0]))
        if doc_year is None:
            return False
        return doc_year <= cutoff_year

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
            fg = self._make_categorical_group(
                feature_key="task",
                field_key="task_field",
                pref=task_pref,
                base_weight=self.feature_weight_groups["essential"].get("task", 8.0),
                force_priority=task_priority,
            )
            if fg.include and fg.fields:
                groups.append(fg)

        task_alias_pref = getattr(essential, "task_aliases", None)
        if task_alias_pref and getattr(task_alias_pref, "include", None):
            alias_priority = self._safe_pref_priority(task_alias_pref, "strong_prefer")
            fg = self._make_categorical_group(
                feature_key="task_alias",
                field_key="task_field",
                pref=task_alias_pref,
                base_weight=self.feature_weight_groups["essential"].get("task", 8.0) * 0.85,
                force_priority=alias_priority,
            )
            if fg.include and fg.fields:
                groups.append(fg)

        for key, field_key in [
            ("domain", "domain_field"),
            ("author", "author_field"),
            ("objective", "objective_field"),
        ]:
            pref = getattr(essential, key, None)
            if pref and getattr(pref, "include", None):
                fg = self._make_categorical_group(
                    feature_key=key,
                    field_key=field_key,
                    pref=pref,
                    base_weight=self.feature_weight_groups["essential"].get(key, 1.0),
                    force_priority=self._safe_pref_priority(pref),
                )
                if fg.include and fg.fields:
                    groups.append(fg)

        domain_alias_pref = getattr(essential, "domain_aliases", None)
        if domain_alias_pref and getattr(domain_alias_pref, "include", None):
            alias_priority = self._safe_pref_priority(domain_alias_pref, "strong_prefer")
            fg = self._make_categorical_group(
                feature_key="domain_alias",
                field_key="domain_field",
                pref=domain_alias_pref,
                base_weight=self.feature_weight_groups["essential"].get("domain", 2.0) * 0.9,
                force_priority=alias_priority,
            )
            if fg.include and fg.fields:
                groups.append(fg)

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
                fg = self._make_categorical_group(
                    feature_key=key,
                    field_key=field_key,
                    pref=pref,
                    base_weight=self.feature_weight_groups["preference"].get(key, 1.0),
                )
                if fg.include or fg.exclude:
                    if fg.fields:
                        groups.append(fg)

        last_modified_pref = self._extract_last_modified_pref(features)
        if last_modified_pref and getattr(last_modified_pref, "include", None):
            cutoff_year = _coerce_year((last_modified_pref.include or [None])[0])
            if cutoff_year is not None:
                groups.append(self._make_direct_group(
                    feature_key="last_modified_year",
                    include=[str(cutoff_year)],
                    fields=self._fields("last_modified_field"),
                    priority="must",
                    base_weight=0.0,
                ))

        functional = getattr(features, "functional", None)
        functional_items = list(getattr(functional, "F_features", None) or [])
        functional_fields = self._functional_feature_fields()
        if functional_items and functional_fields:
            candidates_sources = [s.partition(".")[2] or s for s in functional_fields]
            grams_by_value, syn_by_value = self._prepare_group_expansions(
                include=functional_items,
                candidates_sources=candidates_sources,
                feature_key="functional",
                apply_synonym_confidence_threshold=True,
            )

            for idx, item in enumerate(functional_items):
                groups.append(FeatureGroup(
                    feature_key="functional",
                    priority="prefer",
                    include=[item],
                    exclude=[],
                    fields=functional_fields,
                    grams_by_value={item: grams_by_value.get(item, [])},
                    syn_by_value={item: syn_by_value.get(item, [])},
                    base_weight=self.feature_weight_groups["functional"].get("functional_item", 1.0),
                ))

        quality = getattr(features, "quality", None)
        if quality is not None:
            for attr_name, field_path, weight in self.quality_feature_specs:
                raw_value = getattr(quality, attr_name, None)
                if raw_value is None:
                    continue
                groups.append(self._make_direct_group(
                    feature_key=attr_name,
                    include=[raw_value],
                    fields=[field_path],
                    priority="prefer",
                    base_weight=weight,
                ))

        return groups

    def _prepare_group_expansions(
        self,
        *,
        include: List[Any],
        candidates_sources: List[str],
        feature_key: str,
        apply_synonym_confidence_threshold: bool = True,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[Tuple[str, float]]]]:
        grams_by_value: Dict[str, List[str]] = {}
        syn_by_value: Dict[str, List[Tuple[str, float]]] = {}

        if not include:
            return grams_by_value, syn_by_value

        grams_by_value = self._resolve_grams_grouped_cached(
            candidates_sources,
            include,
            feature_key=feature_key,
        )
        syn_by_value = self._resolve_syns_grouped_cached(
            candidates_sources,
            include,
            feature_key=feature_key,
        )

        for value in include:
            base_variants = _text_variants(value)
            existing_grams = list(grams_by_value.get(value, []))
            existing_grams.extend(x for x in base_variants if _normalize_value_key(x) != _normalize_value_key(value))
            grams_by_value[value] = _dedupe_values_preserve_order(existing_grams)

            existing_syns = list(syn_by_value.get(value, []))
            seen_syn_keys = {_normalize_value_key(s) for s, _ in existing_syns}
            for variant in base_variants:
                variant_key = _normalize_value_key(variant)
                if variant_key and variant_key != _normalize_value_key(value) and variant_key not in seen_syn_keys:
                    existing_syns.append((variant, 0.97))
                    seen_syn_keys.add(variant_key)

            if apply_synonym_confidence_threshold:
                existing_syns = [
                    (s, float(w)) for (s, w) in existing_syns
                    if float(w) >= self.synonym_min_conf
                ]

            syn_by_value[value] = _dedupe_scored_terms_preserve_best(existing_syns)

        return grams_by_value, syn_by_value

    def _make_categorical_group(
        self,
        *,
        feature_key: str,
        field_key: str,
        pref: Any,
        base_weight: float,
        force_priority: Optional[PreferencePriority] = None,
    ) -> FeatureGroup:
        pr: PreferencePriority = force_priority or getattr(pref, "priority", "prefer")
        include = _dedupe_values_preserve_order(list(getattr(pref, "include", None) or []))
        _exclude_ignored = _dedupe_values_preserve_order(list(getattr(pref, "exclude", None) or []))
        fields = self._fields(field_key)
        if not fields:
            return FeatureGroup(
                feature_key=feature_key,
                priority=pr,
                include=[],
                exclude=[],
                fields=[],
                grams_by_value={},
                syn_by_value={},
                base_weight=float(base_weight),
            )

        candidates_sources = [s.partition(".")[2] or s for s in fields]

        grams_by_value, syn_by_value = self._prepare_group_expansions(
            include=include,
            candidates_sources=candidates_sources,
            feature_key=feature_key,
            apply_synonym_confidence_threshold=True,
        )


        return FeatureGroup(
            feature_key=feature_key,
            priority=pr,
            include=include,
            exclude=_exclude_ignored,
            fields=fields,
            grams_by_value=grams_by_value,
            syn_by_value=syn_by_value,
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
        exclude: Optional[List[Any]] = None,
    ) -> FeatureGroup:
        return FeatureGroup(
            feature_key=feature_key,
            priority=priority,
            include=_dedupe_values_preserve_order(list(include or [])),
            exclude=_dedupe_values_preserve_order(list(exclude or [])),
            fields=list(fields),
            grams_by_value={},
            syn_by_value={},
            base_weight=float(base_weight),
        )

    # SCORE REASONING


    def _collect_needed_source_paths(self, groups: List[FeatureGroup]) -> List[str]:
        paths: List[str] = []
        for fg in groups:
            paths.extend(fg.fields)

        for _, field_path, _, _, _ in self.rank_signal_specs:
            paths.append(field_path)

        seen = set()
        out = []
        for p in paths:
            if p and p not in seen:
                out.append(p)
                seen.add(p)
        return out
    


    def _doc_has_term_in_any_field(self, source: Dict[str, Any], fields: List[str], term: str) -> bool:
        term_variants = set(_canonical_compare_variants(term))
        term_variants.update(_normalize_text(v) for v in _text_variants(term))
        term_key = _normalize_value_key(term)
        if term_key:
            term_variants.add(term_key)
        term_variants = {v for v in term_variants if v}

        for f in fields:
            v = _get_by_dotted_path(source, f)
            vals = _as_list(v)
            for candidate in vals:
                candidate_variants = set(_canonical_compare_variants(candidate))
                candidate_variants.update(_normalize_text(v) for v in _text_variants(candidate))
                candidate_key = _normalize_value_key(candidate)
                if candidate_key:
                    candidate_variants.add(candidate_key)
                candidate_variants = {v for v in candidate_variants if v}
                if candidate_variants & term_variants:
                    return True
        return False


    def build_query(self, groups: List[FeatureGroup], *, include_explain: bool = False) -> Dict[str, Any]:
        """
        Returns an ES query dict using should-only positive matching.
        """
        should = self._global_value_shoulds(groups)
        filter_clauses: List[Dict[str, Any]] = []
        for fg in groups:
            clause = self._last_modified_filter_clause(fg)
            if clause is not None:
                filter_clauses.append(clause)

        if should:
            bool_query: Dict[str, Any] = {
                "bool": {
                    "should": should,
                    "minimum_should_match": int(max(0, self.minimum_should_match)),
                }
            }
            if filter_clauses:
                bool_query["bool"]["filter"] = filter_clauses
        elif filter_clauses:
            bool_query = {"bool": {"filter": filter_clauses}}
        else:
            bool_query = {"match_all": {}}

        if self.enable_rank_functions and self.rank_functions:
            rank_query: Dict[str, Any] = {
                "function_score": {
                    "query": bool_query,
                    "score_mode": "sum",
                    "boost_mode": "sum",
                    "functions": self.rank_functions,
                    "max_boost": self.rank_max_boost,
                }
            }
        else:
            rank_query = bool_query

        base_query = rank_query

        docvalue_fields = [field_path for _, field_path, _, _, _ in self.rank_signal_specs]

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
            if not fg.include or fg.priority == "avoid" or fg.feature_key == "last_modified_year":
                continue
            base = self._priority_weight(fg.priority, fg.base_weight)
            if self._is_quality_feature(fg.feature_key):
                for val in fg.include:
                    key = self._value_key(val)
                    if key not in best_by_value or base > best_by_value[key]:
                        best_by_value[key] = float(base)
                continue
            candidate_max = max(
                float(self.MATCH_TYPE_BOOSTS["canonical"]),
                float(self.grams_factor),
                float(self.MATCH_TYPE_BOOSTS["synonyms"]),
            )
            for val in fg.include:
                key = self._value_key(val)
                score = base * candidate_max
                if key not in best_by_value or score > best_by_value[key]:
                    best_by_value[key] = float(score)
        return float(sum(best_by_value.values()))

        
    def compare_bundle_to_sample(
        self,
        features: Any,
        sample_file: Union[str, os.PathLike, Dict[str, Any]],
        prebuilt_groups: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compare a feature bundle against a sample document and return the
        matched items with weights and match labels.
        """
        groups = prebuilt_groups if prebuilt_groups is not None else self.build_feature_groups(features)

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

        hard_filter_results: List[Dict[str, Any]] = []
        hard_filters_passed = True

        for fg in groups:
            if not fg.include or fg.priority == "avoid":
                continue
            if fg.feature_key == "last_modified_year":
                cutoff_year = self._last_modified_cutoff_year(fg)
                doc_value = _get_by_dotted_path(sample_doc, fg.fields[0]) if fg.fields else None
                doc_year = _extract_year_from_datetime_like(doc_value)
                passed = self._doc_satisfies_last_modified(sample_doc, fg)
                hard_filter_results.append({
                    "feature_key": fg.feature_key,
                    "cutoff_year": cutoff_year,
                    "doc_value": doc_value,
                    "doc_year": doc_year,
                    "passed": passed,
                })
                hard_filters_passed = hard_filters_passed and passed
                continue

            feat_weight = self._priority_weight(fg.priority, fg.base_weight)
            feature_matches: List[Dict[str, Any]] = []

            for user_value in fg.include:
                score, detail = self._best_match_for_value_detail(
                    fg, sample_doc, user_value, feat_weight
                )
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
            "hard_filters": hard_filter_results,
            "hard_filters_passed": hard_filters_passed,
            "excluded_by_hard_filters": not hard_filters_passed,
            "total_match_score": total_score if hard_filters_passed else 0.0,
            "normalized_match_score": min(100.0, (total_score / max_score) * 100.0) if hard_filters_passed else 0.0,
            "match_type_boosts": dict(self.MATCH_TYPE_BOOSTS),
            "minimum_should_match": int(self.minimum_should_match),
        }
    
    def precompute_feature_group_cache(self, features: Any) -> List[FeatureGroup]:
        """
        Force synonym/gram computation once, so repeated experiment configs can
        reuse cached AliasResolver results instead of recomputing them.
        """
        return self.build_feature_groups(features)

    def _ensure_dict(self, resp):
        # Elasticsearch 8.x returns ObjectApiResponse
        if hasattr(resp, "body"):
            return resp.body
        return resp
    
    def search(self, es_client: Any, index: str, features: Any, *, include_explain: bool = False, prebuilt_groups: Optional[List[FeatureGroup]] = None, ):
        groups = prebuilt_groups if prebuilt_groups is not None else self.build_feature_groups(features)
        fixed_max_score = max(1e-6, float(self.compute_max_score(groups)))

        q = self.build_query(groups, include_explain=include_explain)
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

        if not include_explain:
            for h in hits_list:
                h.pop("_explanation", None)

        return resp_dict, q, groups
