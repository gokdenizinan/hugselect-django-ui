from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal, Union
from collections import defaultdict
import math


PreferencePriority = Literal["must", "strong_prefer", "prefer", "avoid"]
RelaxLevel = Literal[0, 1, 2, 3, 4]
MappingValue = Union[str, List[str]]
MappingType = Dict[str, MappingValue]


@dataclass
class FeatureGroup:
    feature_key: str
    priority: PreferencePriority
    include: List[Any] = field(default_factory=list)
    exclude: List[Any] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)

    # retained only for compatibility with old code
    grams_by_value: Dict[str, List[str]] = field(default_factory=dict)
    syn_by_value: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    level: int = 0
    relaxable: bool = True
    base_weight: float = 1.0
    per_value_weight: Optional[float] = None


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
    per_value_groups = [
        {"bool": {"should": [{"term": {f: v}} for f in fields], "minimum_should_match": 1}}
        for v in values
    ]
    return {"bool": {"should": per_value_groups, "minimum_should_match": int(k)}}


def _terms_any(fields: List[str], values: List[str]) -> Dict[str, Any]:
    return {"bool": {"should": [{"terms": {f: values}} for f in fields], "minimum_should_match": 1}}


def _dis_max_once(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"dis_max": {"tie_breaker": 0.0, "queries": queries}}


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]


def _get_by_dotted_path(obj: Dict[str, Any], path: str) -> Any:
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


def _quality_attr_to_mapping_key(attr_name: str) -> str:
    return f"{attr_name.lower()}_field"


def _quality_attr_to_default_path(attr_name: str) -> str:
    return f"Quality.{attr_name.replace('_', ' ')}.score"


class ESQueryBuilderAdaptive:
    """
    Drop-in compatibility version of the original class, but simplified to
    a straightforward BM25-style query builder.

    Public API is intentionally preserved so existing callers keep working.
    Many legacy parameters are accepted but unused.
    """

    LEVEL_PENALTY: Dict[int, float] = {
        0: 1.00,
        1: 1.00,
        2: 1.00,
        3: 1.00,
        4: 0.00,
    }

    SCORE_BREAKDOWN_TOLERANCE = 1e-6

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
        synonym_min_conf: float = 0.35,
        max_synonyms_per_value: int = 10,
        tier_boost_start: float = 100.0,
        tier_boost_step: float = 100.0,
        rank_max_boost: float = 50.0,
    ):
        self.mapping = mapping
        self.size = size
        self.target_hits = target_hits
        self.max_relax_steps = max_relax_steps
        self.synonym_min_conf = float(synonym_min_conf)
        self.max_synonyms_per_value = int(max_synonyms_per_value)
        self.tier_boost_start = float(tier_boost_start)
        self.tier_boost_step = float(tier_boost_step)
        self.rank_max_boost = float(rank_max_boost)

        self.priority_multipliers = priority_multipliers or {
            "must": 2.5,
            "strong_prefer": 1.75,
            "prefer": 1.0,
            "avoid": 0.0,
        }

        default_feature_weight_groups: Dict[str, Dict[str, float]] = {
            "essential": {
                "task": 6.0,
                "domain": 3.0,
                "author": 2.5,
                "objective": 3.0,
            },
            "preference": {
                "license_name": 1.5,
                "library_name": 1.5,
                "basemodels": 1.2,
                "datasets": 1.2,
                "language": 1.2,
                "metrics": 1.0,
            },
            "functional": {
                "functional_item": 2.0,
            },
            "quality": {
                "Functional_Suitability": 0.0,
                "Compatibility": 0.0,
                "Performance_Efficiency": 0.0,
                "Reliability": 0.0,
                "Interaction_Capability": 0.0,
                "Security": 0.0,
                "Maintainability": 0.0,
                "Flexibility": 0.0,
            },
            "rank": {
                "likes": 0.15,
                "downloads_last_30_days": 0.20,
                "Functional_Suitability": 0.0,
                "Compatibility": 0.0,
                "Performance_Efficiency": 0.0,
                "Reliability": 0.0,
                "Interaction_Capability": 0.0,
                "Security": 0.0,
                "Maintainability": 0.0,
                "Flexibility": 0.0,
            },
        }

        if feature_weight_groups:
            for section, weights in feature_weight_groups.items():
                default_feature_weight_groups.setdefault(section, {}).update(weights)

        self.feature_weight_groups = default_feature_weight_groups
        self.base_signal_weights = base_signal_weights or {}

        self.boost_config = boost_config or {}
        self.grams_factor = 1.0  # kept for compatibility

        self.rank_field_weights = dict(self.feature_weight_groups.get("rank", {}))
        self.rank_functions: List[Dict[str, Any]] = self._default_rank_functions()

    # ----------------------------
    # Mapping helpers
    # ----------------------------

    def _fields(self, field_key: str) -> List[str]:
        v = self.mapping.get(field_key)
        if v is None:
            return []
        return [v] if isinstance(v, str) else list(v)

    def _first_field(self, field_key: str) -> str:
        fields = self._fields(field_key)
        if not fields:
            raise KeyError(f"Missing mapping for field_key={field_key!r}")
        return fields[0]

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

        return specs

    # ----------------------------
    # Compatibility helpers
    # ----------------------------

    def _flat_grams(self, fg: FeatureGroup) -> List[str]:
        out: List[str] = []
        for v in fg.include:
            out.extend(fg.grams_by_value.get(v, []))
        return list(dict.fromkeys(out))

    def _flat_syns(self, fg: FeatureGroup) -> List[str]:
        out: List[str] = []
        for v in fg.include:
            for s, _ in fg.syn_by_value.get(v, []):
                out.append(s)
        return list(dict.fromkeys(out))

    def _add_value_candidates(
        self,
        *,
        value_key: str,
        value_queries: Dict[str, List[Dict[str, Any]]],
        fg: FeatureGroup,
        val: str,
        feat_weight: float,
        mode: Literal["best", "canonical_only", "grams_only", "synonyms_only"],
        grams_factor: Optional[float] = None,
    ):
        # retained for compatibility; simplified behavior
        if feat_weight <= 0:
            return
        value_queries[value_key].append(
            _wrap_constant_score(_terms_many(fg.fields, [val], k=1), feat_weight)
        )

    def _global_value_shoulds(self, groups: List[FeatureGroup]) -> List[Dict[str, Any]]:
        shoulds: List[Dict[str, Any]] = []
        for fg in groups:
            if not fg.include or fg.priority == "avoid":
                continue
            for val in fg.include:
                weight = self._priority_weight(fg.priority, fg.base_weight)
                shoulds.append(
                    _wrap_constant_score(_terms_many(fg.fields, [str(val)], k=1), weight)
                )
        return shoulds

    # ----------------------------
    # Rank functions
    # ----------------------------

    def _default_rank_functions(self) -> List[Dict[str, Any]]:
        functions: List[Dict[str, Any]] = []
        for _, field_path, modifier, missing, weight in self._rank_signal_specs():
            body: Dict[str, Any] = {"field": field_path, "missing": float(missing)}
            if modifier:
                body["modifier"] = modifier
            functions.append({"field_value_factor": body, "weight": float(weight)})
        return functions

    # ----------------------------
    # FeatureGroup builders
    # ----------------------------

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

        return FeatureGroup(
            feature_key=feature_key,
            priority=pr,
            include=include,
            exclude=exclude,
            fields=fields,
            grams_by_value={},
            syn_by_value={},
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

    def build_feature_groups(self, features: Any) -> List[FeatureGroup]:
        groups: List[FeatureGroup] = []

        task_pref = getattr(getattr(features, "essential", None), "task", None)
        if task_pref and getattr(task_pref, "include", None):
            groups.append(self._make_categorical_group(
                feature_key="task",
                field_key="task_field",
                pref=task_pref,
                base_weight=self.feature_weight_groups["essential"].get("task", 6.0),
                relaxable=False,
                force_priority="must",
            ))

        for key, field_key in [
            ("domain", "domain_field"),
            ("author", "author_field"),
            ("objective", "objective_field"),
        ]:
            pref = getattr(getattr(features, "essential", None), key, None)
            if pref and (getattr(pref, "include", None) or getattr(pref, "exclude", None)):
                groups.append(self._make_categorical_group(
                    feature_key=key,
                    field_key=field_key,
                    pref=pref,
                    base_weight=self.feature_weight_groups["essential"].get(key, 1.0),
                    relaxable=True,
                ))

        for key, field_key in [
            ("license_name", "license_field"),
            ("library_name", "library_name_field"),
            ("basemodels", "basemodels_field"),
            ("datasets", "datasets_field"),
            ("language", "language_field"),
            ("metrics", "metrics_field"),
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
                    base_weight=self.feature_weight_groups["functional"].get("functional_item", 2.0),
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

    # ----------------------------
    # BM25 query construction
    # ----------------------------

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

    def _append_multi_match(
        self,
        should: List[Dict[str, Any]],
        query_text: str,
        fields: List[str],
        boost: float,
        operator: str = "or",
    ) -> None:
        if not query_text or not fields:
            return
        should.append({
            "multi_match": {
                "query": str(query_text),
                "fields": fields,
                "type": "best_fields",
                "operator": operator,
                "boost": float(boost),
            }
        })

    def _append_exact_filter(
        self,
        filters: List[Dict[str, Any]],
        fields: List[str],
        values: List[Any],
    ) -> None:
        if not fields or not values:
            return

        should_clauses: List[Dict[str, Any]] = []
        str_values = [str(v) for v in values]
        for field in fields:
            should_clauses.append({"terms": {field: str_values}})

        filters.append({
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
            }
        })

    def build_tier_filter(self, groups: List[FeatureGroup]) -> Dict[str, Any]:
        # compatibility only; no tier logic now
        return {"match_all": {}}

    def build_query(
        self,
        groups: List[FeatureGroup],
        *,
        include_explain: bool = False,
        tier_filters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        filters: List[Dict[str, Any]] = []
        must_not: List[Dict[str, Any]] = []
        should: List[Dict[str, Any]] = []

        bm25_fields = [
            "Features^4",
            "Metadata.tags^3",
            "Metadata.pipeline_tag^4",
            "Metadata.model_type^2",
            "Metadata.library_name^2",
            "Metadata.basemodels^2",
            "Metadata.datasets^2",
            "Metadata.language^1.5",
            "Metadata.metrics^1.5",
            "author^2",
            "model_id^2",
        ]

        for fg in groups:
            if not fg.include and not fg.exclude:
                continue

            boost = self._priority_weight(fg.priority, fg.base_weight)

            if fg.priority == "avoid":
                for val in fg.exclude or fg.include:
                    if fg.fields:
                        must_not.append(_terms_any(fg.fields, [str(val)]))
                continue

            # only task is kept as hard filter by default
            if fg.feature_key == "task" and fg.include and fg.fields:
                self._append_exact_filter(filters, fg.fields, fg.include)

            for val in fg.include:
                self._append_multi_match(
                    should=should,
                    query_text=str(val),
                    fields=bm25_fields,
                    boost=boost,
                    operator="or",
                )

            for val in fg.exclude:
                if fg.fields:
                    must_not.append(_terms_any(fg.fields, [str(val)]))

        bool_query: Dict[str, Any] = {
            "bool": {
                "filter": filters,
                "should": should,
                "must_not": must_not,
                "minimum_should_match": 1 if should else 0,
            }
        }

        docvalue_fields = [field_path for _, field_path, _, _, _ in self._rank_signal_specs()]

        if self.rank_functions:
            query_body: Dict[str, Any] = {
                "function_score": {
                    "query": bool_query,
                    "score_mode": "sum",
                    "boost_mode": "sum",
                    "functions": self.rank_functions,
                    "max_boost": self.rank_max_boost,
                }
            }
        else:
            query_body = bool_query

        return {
            "explain": bool(include_explain),
            "size": self.size,
            "_source": {"includes": self._collect_needed_source_paths(groups)},
            "docvalue_fields": docvalue_fields,
            "query": query_body,
            "sort": [{"_score": {"order": "desc"}}],
        }

    # ----------------------------
    # Compatibility scoring methods
    # ----------------------------

    def compute_max_score(self, groups: List[FeatureGroup]) -> float:
        total = 0.0
        for fg in groups:
            if fg.priority == "avoid":
                continue
            total += len(fg.include) * self._priority_weight(fg.priority, fg.base_weight)
        return max(1.0, float(total))

    def _get_docvalue(self, hit: Dict[str, Any], field_path: str) -> Any:
        fields = hit.get("fields") or {}
        v = fields.get(field_path)
        if isinstance(v, list):
            return v[0] if v else None
        return v

    def _rank_function_contribs_from_hit(self, hit: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
        breakdown: List[Dict[str, Any]] = []
        total = 0.0

        for signal_name, field_path, modifier, missing, weight in self._rank_signal_specs():
            raw = self._get_docvalue(hit, field_path)
            if raw is None:
                source = hit.get("_source", {}) or {}
                raw = _get_by_dotted_path(source, field_path)

            val = float(raw) if raw is not None else float(missing)
            modded = math.log1p(max(0.0, val)) if modifier == "log1p" else val
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

        return total, breakdown

    def _doc_has_term_in_any_field(self, source: Dict[str, Any], fields: List[str], term: str) -> bool:
        term_norm = str(term).strip().lower()
        for f in fields:
            v = _get_by_dotted_path(source, f)
            vals = _as_list(v)
            for x in vals:
                sx = str(x).strip().lower()
                if term_norm == sx or term_norm in sx:
                    return True
        return False

    def _best_match_for_value(
        self,
        fg: FeatureGroup,
        source: Dict[str, Any],
        user_value: str,
        feat_weight: float,
        *,
        grams_factor: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        matched = self._doc_has_term_in_any_field(source, fg.fields, user_value)
        if matched:
            return feat_weight, {
                "match_type": "canonical",
                "matched_term": user_value,
                "factor": 1.0,
                "score": feat_weight,
            }
        return 0.0, {
            "match_type": "none",
            "matched_term": None,
            "factor": 0.0,
            "score": 0.0,
        }

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
        score, _ = self._best_match_for_value(fg, source, user_value, feat_weight, grams_factor=grams_factor)
        return score

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
            for user_value in fg.include:
                value_key = self._value_key(user_value)
                score = self._best_match_for_value_mode(fg, source, str(user_value), feat_weight, "best")
                if score > best_score_by_value[value_key]:
                    best_score_by_value[value_key] = score
                    winners[value_key] = {
                        "user_value": user_value,
                        "feature_key": fg.feature_key,
                        "priority": fg.priority,
                        "level": int(fg.level),
                        "effective_feat_weight": feat_weight,
                        "mode": "best",
                        "score": float(score),
                    }

        total = float(sum(best_score_by_value.values()))
        return total, winners

    def display_score_for_hit(
        self,
        groups: List[FeatureGroup],
        hit: Dict[str, Any],
        *,
        fixed_max_score: float,
    ) -> float:
        source = hit.get("_source", {}) or {}
        feature_total, _ = self._feature_total_global_by_value(groups, source)
        denom = max(1e-6, float(fixed_max_score))
        return float(min(100.0, (feature_total / denom) * 100.0))

    def score_breakdown_for_hit(
        self,
        groups: List[FeatureGroup],
        hit: Dict[str, Any],
        *,
        fixed_max_score: float,
    ) -> Dict[str, Any]:
        source = hit.get("_source", {}) or {}
        es_score = float(hit.get("_score") or 0.0)

        feature_rows: List[Dict[str, Any]] = []
        feature_total_global, winners = self._feature_total_global_by_value(groups, source)

        for fg in groups:
            if not fg.include or fg.priority == "avoid":
                continue

            feat_weight = self._priority_weight(fg.priority, fg.base_weight)
            per_value: List[Dict[str, Any]] = []
            feat_sum = 0.0

            for user_value in fg.include:
                best_score, best_detail = self._best_match_for_value(
                    fg, source, str(user_value), feat_weight
                )
                feat_sum += best_score
                per_value.append({
                    "user_value": user_value,
                    "best": best_detail,
                })

            feature_rows.append({
                "feature_key": fg.feature_key,
                "priority": fg.priority,
                "level": int(fg.level),
                "multipliers": {
                    "base_weight": float(fg.base_weight),
                    "priority_multiplier": float(self.priority_multipliers.get(fg.priority, 1.0)),
                    "level_penalty": 1.0,
                },
                "effective_feat_weight": feat_weight,
                "per_value_dismax": per_value,
                "feature_contribution": feat_sum,
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
                "estimation": "Compatibility-mode score breakdown for simplified BM25 query builder.",
            }
        }

    # ----------------------------
    # Compatibility no-op relaxation methods
    # ----------------------------

    def _base_constraints_query(self, groups: List[FeatureGroup]) -> Dict[str, Any]:
        return {"match_all": {}}

    def feasibility_counts(
        self,
        es_client: Any,
        index: str,
        groups: List[FeatureGroup],
    ) -> Dict[str, Dict[str, int]]:
        # no staged relaxation anymore; preserve return type only
        out: Dict[str, Dict[str, int]] = {}
        for fg in groups:
            out[fg.feature_key] = {
                "strict": 1,
                "grams": 1,
                "syn": 1,
            }
        return out

    def pick_next_relaxation(
        self,
        groups: List[FeatureGroup],
        feas: Dict[str, Dict[str, int]],
    ) -> Optional[FeatureGroup]:
        return None

    def _pick_last_resort(self, groups: List[FeatureGroup]) -> Optional[FeatureGroup]:
        return None

    # ----------------------------
    # Search
    # ----------------------------

    def _ensure_dict(self, resp):
        if hasattr(resp, "body"):
            return resp.body
        return resp

    def search(
        self,
        es_client: Any,
        index: str,
        features: Any,
        *,
        include_score_breakdown: bool = False,
        include_explain: bool = False,
    ):
        groups = self.build_feature_groups(features)
        fixed_max_score = self.compute_max_score(groups)
        fixed_max_score = max(1e-6, float(fixed_max_score))

        q = self.build_query(
            groups,
            include_explain=include_explain,
            tier_filters=[],
        )

        resp = es_client.search(index=index, body=q)
        resp_dict = self._ensure_dict(resp)

        hits_list = resp_dict.get("hits", {}).get("hits", [])
        for i, h in enumerate(hits_list):
            raw_id = h.get("_id")
            pretty_id = normalize_es_id(raw_id) if raw_id else None
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

            if include_score_breakdown:
                new_hit["_score_breakdown"] = self.score_breakdown_for_hit(
                    groups,
                    h,
                    fixed_max_score=fixed_max_score,
                )
            if not include_explain:
                new_hit.pop("_explanation", None)

            hits_list[i] = new_hit

        return resp_dict, q, groups