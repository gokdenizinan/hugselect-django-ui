from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal, Union, Iterable

from datetime import datetime, timedelta
from collections import defaultdict


# ----------------------------
# Types
# ----------------------------
query_mapping = {
    # high-level intent
    "task_field": ["Metadata.pipeline_tag", "Metadata.tags", "Metadata.model_type"],   # e.g. "text-classification"
    "domain_field": ["Metadata.tags", "Features"],        # domain as tags (e.g. "medical", "finance")
    "model_name_field": ["model_id"],
    "author_field": ["author"],
    "objective_field": ["Metadata.tags", "Features"],      # you can reserve special tags for objectives

    # metadata / constraints
    "license_field": ["Metadata.license", "Metadata.tags"],
    "downloads_all_time_field": "Metadata.downloads_all_time",
    "likes_field": "Metadata.likes",
    "downloads_30d_field": "Metadata.downloads_last_30_days",
    "file_count_field": "Metadata.file_count",
    "gated_field": "Metadata.gated",                 # keyword: "true"/"false" or similar
    "library_name_field": ["Metadata.library_name"],
    "model_type_field": ["Metadata.model_type"],
    "basemodels_field": ["Metadata.basemodels"],
    "datasets_field": ["Metadata.datasets"],
    "tensors_total_field": "Metadata.tensors_total",
    "used_storage_field": "Metadata.usedStorage",
    "last_modified_field": "Metadata.lastModified",  # date
    "language_field": ["Metadata.language","Metadata.tags"],
    "metrics_field": ["Metadata.metrics"],   


    # quality dimensions (all are *.score)
    "functional_suitability_field": "Quality.Functional Suitability.score",
    "compatibility_field": "Quality.Compatibility.score",
    "performance_efficiency_field": "Quality.Performance Efficiency.score",
    "reliability_field": "Quality.Reliability.score",
    "interaction_capability_field": "Quality.Interaction Capability.score",
    "security_field": "Quality.Security.score",
    "maintainability_field": "Quality.Maintainability.score",
    "flexibility_field": "Quality.Flexibility.score",
}


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
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)

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


# ----------------------------
# Core: Relaxation-driven ES Query Builder
# ----------------------------

class ESQueryBuilderAdaptive:
    """
    Implements:
      1) Relax feature-by-feature (one feature per stage)
      2) Relax only features with no matches (feasibility tests)
      3) First relaxed check: synonyms (strict is canonical + grams)
      4) When must->should: grams first, then synonyms
      5) Relax order: strong_prefer -> must (prefer already should)
      6) Later-stage matches score less (level penalty)
      7) Normalize final score to 0..100 based on max possible from given query features
      8) No double counting per user value across overlapping categories/fields (dis_max tie_breaker=0)
    """

    LEVEL_PENALTY: Dict[int, float] = {
        0: 1.00,  # strict
        1: 0.85,  # synonym must (still required, less faithful)
        2: 0.70,  # should grams
        3: 0.55,  # should synonyms
        4: 0.00,  # dropped
    }

    def __init__(
        self,
        mapping: MappingType,
        *,
        base_signal_weights: Optional[Dict[str, float]] = None,
        priority_multipliers: Optional[Dict[PreferencePriority, float]] = None,
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

        self.resolver = AliasResolver(
            synonym_provider_factory=make_embedding_provider_factory(min_similarity=0.5),
            max_synonyms=max_synonyms_per_value,
        )

        # feature-weight baselines (you can tune)
        self.base_signal_weights = base_signal_weights or {
            "tag_match": 2.0,
            "author_match": 2.0,
            "task_match": 10.0,
            "domain_match": 1.0,
        }

        self.priority_multipliers = priority_multipliers or {
            "must": 1.6,
            "strong_prefer": 1.3,
            "prefer": 1.0,
            "avoid": 0.0,
        }

        self.synonym_min_conf = float(synonym_min_conf)
        self.max_synonyms_per_value = int(max_synonyms_per_value)

        # ranking signals (optional “global” popularity/quality)
        self.rank_functions: List[Dict[str, Any]] = self._default_rank_functions()
        self.tier_boost_start = float(tier_boost_start)
        self.tier_boost_step = float(tier_boost_step)
        self.rank_max_boost = float(rank_max_boost)

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

    def _add_value_candidates(
        self,
        *,
        value_key: str,
        value_queries: Dict[str, List[Dict[str, Any]]],
        fg: FeatureGroup,
        val: str,
        feat_weight: float,
        mode: Literal["best", "canonical_only", "grams_only", "synonyms_only"],
        grams_factor: float = 0.99,
    ):
        """
        Add scoring candidates for a single (feature group, user value) into the GLOBAL bucket for that value.
        Later we wrap each bucket into one dis_max => value contributes once total.
        """

        if feat_weight <= 0:
            return

        # Canonical
        if mode in ("best", "canonical_only"):
            value_queries[value_key].append(
                _wrap_constant_score(_terms_many(fg.fields, [val], k=1), feat_weight)
            )

        # Grams (per value)
        if mode in ("best", "grams_only"):
            grams_list = fg.grams_by_value.get(val, [])
            if grams_list:
                value_queries[value_key].append(
                    _wrap_constant_score(_terms_many(fg.fields, grams_list, k=1), feat_weight * float(grams_factor))
                )

        # Synonyms (per value, scaled by similarity)
        if mode in ("best", "synonyms_only"):
            syn_list = fg.syn_by_value.get(val, [])
            for syn, sim in syn_list[: self.max_synonyms_per_value]:
                value_queries[value_key].append(
                    _wrap_constant_score(_terms_many(fg.fields, [syn], k=1), feat_weight * float(sim))
                )

    def _global_value_shoulds(self, groups: List[FeatureGroup]) -> List[Dict[str, Any]]:
        """
        Build ONE dis_max per user value across ALL features.
        Prevents cross-feature double counting for the same value.
        """
        value_queries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for fg in groups:
            if not fg.include:
                continue
            if fg.priority == "avoid":
                continue
            if fg.level >= 4:
                continue

            level_pen = float(self.LEVEL_PENALTY.get(int(fg.level), 0.0))
            feat_weight = self._priority_weight(fg.priority, fg.base_weight) * level_pen
            if feat_weight <= 0:
                continue

            # Decide scoring mode for this feature at this level
            # (Keep your intent: must->should: grams first then syns, prefer is "best")
            if fg.priority in ("must", "strong_prefer"):
                if fg.level == 2:
                    mode = "grams_only"
                elif fg.level == 3:
                    mode = "synonyms_only"
                else:
                    mode = "best"
            else:
                mode = "best"

            # Add per-(feature,value) candidates into the GLOBAL per-value bucket
            for val in fg.include:
                # You can normalize this key if you want case-insensitive grouping:
                value_key = val.strip().lower()
                self._add_value_candidates(
                    value_key=value_key,
                    value_queries=value_queries,
                    fg=fg,
                    val=val,
                    feat_weight=float(feat_weight),
                    mode=mode,  # type: ignore[arg-type]
                )

        # Now wrap each user value bucket in dis_max => value contributes once total
        shoulds: List[Dict[str, Any]] = []
        for _, queries in value_queries.items():
            if not queries:
                continue
            shoulds.append(_dis_max_once(queries))

        return shoulds

    from collections import defaultdict

    def display_score_for_hit(
        self,
        groups: List[FeatureGroup],
        hit: Dict[str, Any],
        *,
        fixed_max_score: float,
    ) -> float:
        """
        Feature-only normalized score (0..100), matching ES query semantics:
        - ONE contribution per user value globally (dis_max across all features)
        - Sum those contributions
        """
        source = hit.get("_source", {}) or {}

        best_by_value: Dict[str, float] = defaultdict(float)

        for fg in groups:
            if not fg.include:
                continue
            if fg.priority == "avoid":
                continue
            if fg.level >= 4:
                continue

            level_pen = float(self.LEVEL_PENALTY.get(int(fg.level), 0.0))
            pr_mult = float(self.priority_multipliers.get(fg.priority, 1.0))
            feat_weight = float(fg.base_weight) * pr_mult * level_pen
            if feat_weight <= 0:
                continue

            # Match your _global_value_shoulds() mode selection
            if fg.priority in ("must", "strong_prefer"):
                if fg.level == 2:
                    mode: Literal["best","canonical_only","grams_only","synonyms_only"] = "grams_only"
                elif fg.level == 3:
                    mode = "synonyms_only"
                else:
                    mode = "best"
            else:
                mode = "best"

            for user_value in fg.include:
                # IMPORTANT: normalize same way you do in _global_value_shoulds / compute_max_score
                value_key = user_value.strip().lower()

                score = self._best_match_for_value_mode(
                    fg, source, user_value, feat_weight, mode
                )
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
        """
        Mirrors ES _global_value_shoulds semantics:
        - one bucket per user value (normalized key)
        - each bucket chooses best match across ALL features (dis_max)
        Returns:
        (total, winners_by_value_key)
        """
        best_score_by_value: Dict[str, float] = defaultdict(float)
        winners: Dict[str, Dict[str, Any]] = {}

        for fg in groups:
            if not fg.include or fg.priority == "avoid" or fg.level >= 4:
                continue

            level_pen = float(self.LEVEL_PENALTY.get(int(fg.level), 0.0))
            pr_mult = float(self.priority_multipliers.get(fg.priority, 1.0))
            feat_weight = float(fg.base_weight) * pr_mult * level_pen
            if feat_weight <= 0:
                continue

            # Match your query scoring mode logic
            if fg.priority in ("must", "strong_prefer"):
                if fg.level == 2:
                    mode = "grams_only"
                elif fg.level == 3:
                    mode = "synonyms_only"
                else:
                    mode = "best"
            else:
                mode = "best"

            for user_value in fg.include:
                value_key = user_value.strip().lower()

                # score this fg/value under same mode as ES query
                score = self._best_match_for_value_mode(
                    fg, source, user_value, feat_weight, mode  # <- from earlier fix
                )

                if score > best_score_by_value[value_key]:
                    best_score_by_value[value_key] = score
                    winners[value_key] = {
                        "user_value": user_value,
                        "feature_key": fg.feature_key,
                        "priority": fg.priority,
                        "level": int(fg.level),
                        "effective_feat_weight": feat_weight,
                        "mode": mode,
                        "score": float(score),
                    }

        total = float(sum(best_score_by_value.values()))
        return total, winners
    # ----------------------------
    # Optional rank functions
    # ----------------------------

    def _default_rank_functions(self) -> List[Dict[str, Any]]:
        # You can keep your existing ones; these add non-feature score.
        likes_field = self.mapping.get("likes_field", "Metadata.likes")
        downloads_30d_field = self.mapping.get("downloads_30d_field", "Metadata.downloads_last_30_days")
        quality_field = self.mapping.get("functional_suitability_field", "Quality.Functional Suitability.score")

        return [
            {"field_value_factor": {"field": likes_field, "missing": 0.0, "modifier": "log1p"}, "weight": 0.8},
            {"field_value_factor": {"field": downloads_30d_field, "missing": 0.0, "modifier": "log1p"}, "weight": 1.2},
            {"field_value_factor": {"field": quality_field, "missing": 0.0}, "weight": 1.5},
        ]

    # ----------------------------
    # Build FeatureGroups from your FeatureBundle
    # ----------------------------

    def build_feature_groups(
        self,
        features: Any,  # your FeatureBundle
    ) -> List[FeatureGroup]:
        """
        This assumes your FeatureBundle has:
          features.essential.task, .domain, .author, .objective (categorical-like)
          features.preferences.<...> (categorical/numeric/bool/recency)
        We focus on categorical scoring/relaxation here because your requirements are about categorical matching.
        You can extend numeric/bool/recency similarly (usually they’re filter relaxable).
        """
        groups: List[FeatureGroup] = []

        # --- task (must, not relaxable) ---
        task_pref = getattr(getattr(features, "essential", None), "task", None)
        if task_pref and getattr(task_pref, "include", None):
            fg = self._make_categorical_group(
                feature_key="task",
                field_key="task_field",
                pref=task_pref,
                base_weight=self.base_signal_weights.get("task_match", 10.0),
                relaxable=False,  # IMPORTANT: keep task locked
                force_priority="must",
            )
            groups.append(fg)

        # --- essential domain/author/objective (usually relaxable, but you decide) ---
        for key, field_key, base_w in [
            ("domain", "domain_field", self.base_signal_weights.get("domain_match", 2.0)),
            ("author", "author_field", self.base_signal_weights.get("author_match", 3.0)),
            ("objective", "objective_field", self.base_signal_weights.get("tag_match", 1.0)),
        ]:
            pref = getattr(getattr(features, "essential", None), key, None)
            if pref and getattr(pref, "include", None):
                groups.append(self._make_categorical_group(
                    feature_key=key,
                    field_key=field_key,
                    pref=pref,
                    base_weight=base_w,
                    relaxable=True,
                ))

        # --- preference categorical features you listed ---
        for key, field_key, base_w in [
            ("license_name", "license_field", self.base_signal_weights.get("tag_match", 1.0)),
            ("library_name", "library_name_field", self.base_signal_weights.get("tag_match", 1.0)),
            ("basemodels", "basemodels_field", self.base_signal_weights.get("tag_match", 1.0)),
            ("datasets", "datasets_field", self.base_signal_weights.get("tag_match", 1.0)),
            ("language", "language_field", self.base_signal_weights.get("tag_match", 1.0)),
            ("metrics", "metrics_field", self.base_signal_weights.get("tag_match", 1.0)),
        ]:
            pref = getattr(getattr(features, "preferences", None), key, None)
            if pref and (getattr(pref, "include", None) or getattr(pref, "exclude", None)):
                groups.append(self._make_categorical_group(
                    feature_key=key,
                    field_key=field_key,
                    pref=pref,
                    base_weight=base_w,
                    relaxable=True,
                ))

        # You can append numeric/bool/recency groups here with similar level logic if needed.

        return groups

    def build_tier_filter(self, groups: List[FeatureGroup]) -> Dict[str, Any]:
        """
        Build a query clause representing the *eligibility constraints* of the current tier.
        Used only for tier boosting, not for retrieval.
        """
        flt: List[Dict[str, Any]] = []
        must_not: List[Dict[str, Any]] = []

        # same exclude handling as build_query
        for fg in groups:
            if fg.exclude:
                ex_grams = self.resolver.resolve_grams(
                    [s.partition(".")[2] or s for s in fg.fields],
                    fg.exclude
                )
                if ex_grams:
                    must_not.append(_terms_any(fg.fields, ex_grams))

        # same must/strong_prefer “required constraints” logic as build_query
        for fg in groups:
            if not fg.include:
                continue

            if fg.priority in ("must", "strong_prefer") and fg.level in (0, 1):
                must_variants = [_terms_many(fg.fields, fg.include, k=1)]

                flat_grams: List[str] = []
                for v in fg.include:
                    flat_grams.extend(fg.grams_by_value.get(v, []))
                if flat_grams:
                    flat_grams = list(dict.fromkeys(flat_grams))
                    must_variants.append(_terms_many(fg.fields, flat_grams, k=1))

                if fg.level >= 1:
                    flat_syns: List[str] = []
                    for v in fg.include:
                        for s, _ in fg.syn_by_value.get(v, []):
                            flat_syns.append(s)
                    if flat_syns:
                        flat_syns = list(dict.fromkeys(flat_syns))
                        must_variants.append(_terms_many(fg.fields, flat_syns, k=1))

                flt.append(_bool_should(must_variants, msm=1))

        tier_q: Dict[str, Any] = {"bool": {}}
        if flt:
            tier_q["bool"]["filter"] = flt
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

    # SCORE REASONING

    SCORE_BREAKDOWN_TOLERANCE = 1e-6

    def _collect_needed_source_paths(self, groups: List[FeatureGroup]) -> List[str]:
        """
        Collect all fields that we need to pull from _source to compute breakdowns.
        We include feature fields + rank function fields.
        """
        paths: List[str] = []
        for fg in groups:
            paths.extend(fg.fields)

        # Rank fields used by field_value_factor
        likes_field = self.mapping.get("likes_field", "Metadata.likes")
        downloads_30d_field = self.mapping.get("downloads_30d_field", "Metadata.downloads_last_30_days")
        quality_field = self.mapping.get("functional_suitability_field", "Quality.Functional Suitability.score")
        paths.extend([likes_field, downloads_30d_field, quality_field])

        # de-dupe preserving order
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
        likes_field = self.mapping.get("likes_field", "Metadata.likes")
        downloads_30d_field = self.mapping.get("downloads_30d_field", "Metadata.downloads_last_30_days")
        quality_field = self.mapping.get("functional_suitability_field", "Quality.Functional Suitability.score")

        breakdown: List[Dict[str, Any]] = []
        total = 0.0

        def fvf(field_path: str, weight: float, modifier: Optional[str], missing: float) -> None:
            nonlocal total, breakdown

            raw = self._get_docvalue(hit, field_path)
            # Fallback to _source ONLY if docvalues not returned (optional)
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
                "field": field_path,
                "raw_value": raw,
                "missing_used": (raw is None),
                "modifier": modifier or "none",
                "after_modifier": modded,
                "weight": float(weight),
                "contribution": contrib,
            })

        fvf(likes_field, weight=0.8, modifier="log1p", missing=0.0)
        fvf(downloads_30d_field, weight=1.2, modifier="log1p", missing=0.0)
        fvf(quality_field, weight=1.5, modifier=None, missing=0.0)

        return total, breakdown


    def _doc_has_term_in_any_field(self, source: Dict[str, Any], fields: List[str], term: str) -> bool:
        for f in fields:
            v = _get_by_dotted_path(source, f)
            vals = _as_list(v)
            # normalize to str for comparison if your stored values are strings
            if any(str(x) == str(term) for x in vals):
                return True
        return False

    def _best_match_for_value(
        self,
        fg: FeatureGroup,
        source: Dict[str, Any],
        user_value: str,
        feat_weight: float,
        *,
        grams_factor: float = 0.99,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Recompute the dis_max best-of for ONE user value:
          canonical: feat_weight
          grams: feat_weight * grams_factor (if any gram matches)
          synonyms: feat_weight * sim (best sim among matching syns)
        Return (best_score, detail).
        """
        candidates: List[Tuple[float, Dict[str, Any]]] = []

        # canonical
        if self._doc_has_term_in_any_field(source, fg.fields, user_value):
            candidates.append((feat_weight, {
                "match_type": "canonical",
                "matched_term": user_value,
                "factor": 1.0,
                "score": feat_weight,
            }))

        # grams (note: your current query allows grams from the whole feature, not per value)
        # We'll mirror your current implementation: if ANY gram matches, it can win this value's dis_max.
        # If you later change to grams-per-value, update this accordingly.
        # grams (PER VALUE)
        grams_list = fg.grams_by_value.get(user_value, [])
        for g in grams_list:
            if self._doc_has_term_in_any_field(source, fg.fields, g):
                s = feat_weight * float(grams_factor)
                candidates.append((s, {
                    "match_type": "grams",
                    "matched_term": g,
                    "factor": float(grams_factor),
                    "score": s,
                }))
                break


        # synonyms (your fg.syn_pairs is global — this can double count in ES today, but here we pick best)
        # synonyms (PER VALUE): pick best similarity among synonyms that match this doc
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

    def score_breakdown_for_hit(self,groups: List[FeatureGroup],hit: Dict[str, Any],*,fixed_max_score: float, ) -> Dict[str, Any]:

        """
        Build a full score breakdown for a single ES hit.
        """
        source = hit.get("_source", {}) or {}
        es_score = float(hit.get("_score") or 0.0)

        feature_rows: List[Dict[str, Any]] = []

        for fg in groups:
            if not fg.include:
                continue
            if fg.priority == "avoid":
                continue
            if fg.level >= 4:
                continue

            level_pen = float(self.LEVEL_PENALTY.get(int(fg.level), 0.0))
            pr_mult = float(self.priority_multipliers.get(fg.priority, 1.0))
            base = float(fg.base_weight)

            feat_weight = base * pr_mult * level_pen

            # if weight is 0, it contributes nothing
            per_value: List[Dict[str, Any]] = []
            feat_sum = 0.0

            for user_value in fg.include:
                best_score, best_detail = self._best_match_for_value(
                    fg, source, user_value, feat_weight
                )
                feat_sum += best_score
                per_value.append({
                    "user_value": user_value,
                    "best": best_detail,
                })

            feature_total_global, winners = self._feature_total_global_by_value(groups, source)
            feature_rows.append({
                "feature_key": fg.feature_key,
                "priority": fg.priority,
                "level": int(fg.level),
                "multipliers": {
                    "base_weight": base,
                    "priority_multiplier": pr_mult,
                    "level_penalty": level_pen,
                },
                "effective_feat_weight": feat_weight,
                "per_value_dismax": per_value,
                "feature_contribution": feat_sum,
            })

        # global rank function contributions
        rank_total, rank_rows = self._rank_function_contribs_from_hit(hit)

        raw_est = feature_total_global + rank_total
                
        denom = max(1e-6, float(fixed_max_score))
        feature_score_0_100 = min(100.0, (feature_total_global / denom) * 100.0)


        # # --- STEP 7: sanity check (temporary) ---
        # if abs(raw_est - es_score) > 1e-3:
        #     print(
        #         "WARNING: score breakdown mismatch",
        #         {
        #             "es_score": es_score,
        #             "estimated_raw_score": raw_est,
        #             "feature_total_est": feature_total_global,
        #             "rank_total_est": rank_total,
        #         }
        #     )

        return {
            "es_score": es_score,
            "estimated_raw_score": raw_est,
            "feature_score_0_100": feature_score_0_100,
            "feature_total_est": feature_total_global,          # <- ES-matching
            "value_winners": winners,    
            "rank_total_est": rank_total,
            "features": feature_rows,
            "rank_functions": rank_rows,
            "notes": {
                "estimation": "Computed from _source values using the same multipliers/penalties as query builder. ES _score may differ if docvalues/_source differ, analyzers differ, or if ES scoring differs from our assumptions.",
            }
        }

    # ----------------------------
    # Build ES query from current FeatureGroup states
    # ----------------------------

    def build_query(self, groups: List[FeatureGroup], *, include_explain: bool = False,
                    tier_filters: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:



        """
        Returns an ES query dict.
        Uses:
          - strict constraints (must/filter) per feature level rules
          - should scoring with dis_max per user value to avoid double counting
          - function_score for global ranking signals
          - optional script_score normalization to 0..100
        """

        must: List[Dict[str, Any]] = []
        flt: List[Dict[str, Any]] = []
        should: List[Dict[str, Any]] = []
        must_not: List[Dict[str, Any]] = []

        # 1) Excludes always must_not (safe)
        for fg in groups:
            if fg.exclude:
                # gram-based exclude (safe-ish)
                ex_grams = self.resolver.resolve_grams(
                    [s.partition(".")[2] or s for s in fg.fields],
                    fg.exclude
                )
                if ex_grams:
                    must_not.append(_terms_any(fg.fields, ex_grams))

        # 2) Feature constraints + scoring
        for fg in groups:
            if not fg.include:
                continue

            pr = fg.priority
            if pr == "avoid":
                flat = self._flat_grams(fg)
                if flat:
                    must_not.append(_terms_any(fg.fields, flat))


            # Feature-level weight for scoring (decays by relaxation level)
            level_pen = self.LEVEL_PENALTY.get(int(fg.level), 0.0)

            # --- Required constraints for must/strong_prefer in level 0..1 ---
            if pr in ("must", "strong_prefer") and fg.level in (0, 1):
                must_variants = [_terms_many(fg.fields, fg.include, k=1)]

                # flatten grams across values
                flat_grams: List[str] = []
                for v in fg.include:
                    flat_grams.extend(fg.grams_by_value.get(v, []))
                if flat_grams:
                    flat_grams = list(dict.fromkeys(flat_grams))  # optional dedupe
                    must_variants.append(_terms_many(fg.fields, flat_grams, k=1))

                # flatten synonyms across values (for MUST feasibility only)
                if fg.level >= 1:
                    flat_syns: List[str] = []
                    for v in fg.include:
                        for s, _ in fg.syn_by_value.get(v, []):
                            flat_syns.append(s)
                    if flat_syns:
                        flat_syns = list(dict.fromkeys(flat_syns))  # optional dedupe
                        must_variants.append(_terms_many(fg.fields, flat_syns, k=1))

                flt.append(_bool_should(must_variants, msm=1))
        should = self._global_value_shoulds(groups)

        bool_query: Dict[str, Any] = {"bool": {}}
        if must: bool_query["bool"]["must"] = must
        if flt: bool_query["bool"]["filter"] = flt
        if should: bool_query["bool"]["should"] = should
        if must_not: bool_query["bool"]["must_not"] = must_not
        bool_query["bool"]["minimum_should_match"] = 0

        # inner: rank signals with a hard cap
        rank_query: Dict[str, Any] = {
            "function_score": {
                "query": bool_query,
                "score_mode": "sum",
                "boost_mode": "sum",
                "functions": self.rank_functions,
                "max_boost": self.rank_max_boost,   # NEW: cap likes/downloads/quality impact
            }
        }

        # outer: tier boosts (uncapped or lightly capped)
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


        likes_field = self.mapping.get("likes_field", "Metadata.likes")
        downloads_30d_field = self.mapping.get("downloads_30d_field", "Metadata.downloads_last_30_days")
        quality_field = self.mapping.get("functional_suitability_field", "Quality.Functional Suitability.score")

        query = {
            "explain": bool(include_explain),
            "size": self.size,
            "_source": {"includes": self._collect_needed_source_paths(groups)},
            "docvalue_fields": [likes_field, downloads_30d_field, quality_field],  # <-- ADD THIS
            "query": base_query,
            "sort": [{"_score": {"order": "desc"}}],
        }
        return query
    # ----------------------------
    # Max-score computation (normalized scoring)
    # ----------------------------

    def compute_max_score(self, groups: List[FeatureGroup]) -> float:
        # max contribution per VALUE across all features
        best_by_value: Dict[str, float] = {}

        for fg in groups:
            if not fg.include:
                continue
            if fg.priority == "avoid":
                continue
            if fg.level >= 4:
                continue

            # "max possible from given query features"
            # use strict fidelity baseline (penalty=1.0), and priority multiplier
            base = self._priority_weight(fg.priority, fg.base_weight) * self.LEVEL_PENALTY[0]

            for val in fg.include:
                # if the value appears in multiple features, take the max (global dis_max semantics)
                key = val.strip().lower()
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
        grams_factor: float = 0.99,
    ) -> float:
        """
        Recompute the same candidate set your ES query would create for this fg/value at current mode,
        then return the best (dis_max behavior).
        """
        best = 0.0

        # canonical
        if mode in ("best", "canonical_only"):
            if self._doc_has_term_in_any_field(source, fg.fields, user_value):
                best = max(best, float(feat_weight))

        # grams (PER VALUE)
        if mode in ("best", "grams_only"):
            grams_list = fg.grams_by_value.get(user_value, [])
            if grams_list:
                for g in grams_list:
                    if self._doc_has_term_in_any_field(source, fg.fields, g):
                        best = max(best, float(feat_weight) * float(grams_factor))
                        break

        # synonyms (PER VALUE, scaled)
        if mode in ("best", "synonyms_only"):
            syn_list = fg.syn_by_value.get(user_value, [])
            for syn, sim in syn_list[: self.max_synonyms_per_value]:
                if self._doc_has_term_in_any_field(source, fg.fields, syn):
                    best = max(best, float(feat_weight) * float(sim))

        return float(best)


    def _base_constraints_query(self, groups: List[FeatureGroup]) -> Dict[str, Any]:
        """
        Base constraints that we keep fixed while testing feasibility.
        Usually: task (must, not relaxable) and maybe other non-relaxable essentials.
        """
        must: List[Dict[str, Any]] = []
        must_not: List[Dict[str, Any]] = []

        for fg in groups:
            if not fg.include:
                continue
            if not fg.relaxable and fg.priority in ("must", "strong_prefer"):
                variants = [_terms_many(fg.fields, fg.include, k=1)]

                flat_grams = self._flat_grams(fg)
                if flat_grams:
                    variants.append(_terms_many(fg.fields, flat_grams, k=1))

                if fg.feature_key == "task" and fg.level >= 1:
                    flat_syns = self._flat_syns(fg)
                    if flat_syns:
                        variants.append(_terms_many(fg.fields, flat_syns, k=1))

                must.append(_bool_should(variants, msm=1))


            # keeps your excludes too
            if fg.exclude:
                ex_grams = self.resolver.resolve_grams(
                    [s.partition(".")[2] or s for s in fg.fields],
                    fg.exclude
                )
                if ex_grams:
                    must_not.append(_terms_any(fg.fields, ex_grams))

        q: Dict[str, Any] = {"bool": {}}
        if must:
            q["bool"]["must"] = must
        if must_not:
            q["bool"]["must_not"] = must_not
        q["bool"]["minimum_should_match"] = 0
        return q

    def feasibility_counts(
        self,
        es_client: Any,
        index: str,
        groups: List[FeatureGroup],
    ) -> Dict[str, Dict[str, int]]:
        """
        Returns per-feature counts for strict and synonym feasibility under base constraints.

        Output:
          {
            "license_name": {"strict": 12, "syn": 34, "grams": 10},
            ...
          }
        """
        base_q = self._base_constraints_query(groups)

        # Use msearch for performance (works on ES python client: es.msearch(body=...))
        # We'll do count queries as search with size=0 for compatibility.
        mbody: List[Dict[str, Any]] = []
        keys: List[Tuple[str, str]] = []

        def add_count(feature_key: str, label: str, extra_must: Dict[str, Any]):
            keys.append((feature_key, label))
            mbody.append({"index": index})
            mbody.append({
                "size": 0,
                "track_total_hits": True,
                "query": {
                    "bool": {
                        "must": [base_q, extra_must],
                        "minimum_should_match": 0,
                    }
                }
            })


        for fg in groups:
            if not fg.include:
                continue

            if not fg.relaxable and fg.feature_key != "task":
                continue
            if fg.priority not in ("must", "strong_prefer"):
                continue

            terms = [_terms_many(fg.fields, fg.include, k=1)]
            flat_grams = self._flat_grams(fg)
            if flat_grams:
                terms.append(_terms_many(fg.fields, flat_grams, k=1))
            add_count(fg.feature_key, "strict", _bool_should(terms, msm=1))

            # grams alone (useful for level 2)
            flat_grams = self._flat_grams(fg)
            if flat_grams:
                add_count(fg.feature_key, "grams", _terms_many(fg.fields, flat_grams, k=1))
            else:
                add_count(fg.feature_key, "grams", _terms_many(fg.fields, fg.include, k=1))


            # synonyms (for level 1 and 3)
            syn_terms: List[str] = []
            for v in fg.include:
                for s, _ in fg.syn_by_value.get(v, []):
                    syn_terms.append(s)

            flat_syns = self._flat_syns(fg)
            if flat_syns:
                add_count(fg.feature_key, "syn", _terms_many(fg.fields, flat_syns, k=1))
            else:
                add_count(fg.feature_key, "syn", _terms_many(fg.fields, fg.include, k=1))



        if not mbody:
            return {}

        resp = es_client.msearch(body=mbody)
        out: Dict[str, Dict[str, int]] = {}

        for (feature_key, label), r in zip(keys, resp.get("responses", [])):
            total = r.get("hits", {}).get("total", 0)
            value = total.get("value", total) if isinstance(total, dict) else total
            out.setdefault(feature_key, {})[label] = int(value)

        return out

    # ----------------------------
    # Choose next feature to relax (feature-by-feature)
    # ----------------------------

    def pick_next_relaxation(
        self,
        groups: List[FeatureGroup],
        feas: Dict[str, Dict[str, int]],
    ) -> Optional[FeatureGroup]:
        """
        Pick ONE feature to relax next, prioritizing:
          - only features that have no matches at their *current* required state
          - strong_prefer before must
          - lower base_weight first (cheaper to relax)
        """
        candidates: List[FeatureGroup] = []

        for fg in groups:
            if not fg.relaxable:
                continue
            if fg.priority not in ("must", "strong_prefer"):
                continue
            if fg.level >= 4:
                continue

            f = feas.get(fg.feature_key, {})
            strict_cnt = f.get("strict", 0)
            syn_cnt = f.get("syn", 0)
            grams_cnt = f.get("grams", 0)

            # Determine infeasibility at current level
            infeasible = False
            if fg.level == 0:
                infeasible = (strict_cnt == 0)
            elif fg.level == 1:
                infeasible = (syn_cnt == 0)
            elif fg.level == 2:
                infeasible = (grams_cnt == 0)
            elif fg.level == 3:
                infeasible = (syn_cnt == 0)

            if infeasible:
                candidates.append(fg)

        if not candidates:
            return None

        def sort_key(fg: FeatureGroup):
            # strong_prefer first
            pr_rank = 0 if fg.priority == "strong_prefer" else 1
            return (pr_rank, fg.base_weight)

        candidates.sort(key=sort_key)
        return candidates[0]

    # ----------------------------
    # Main search loop: adaptive relaxation
    # ----------------------------

    def _ensure_dict(self, resp):
        # Elasticsearch 8.x returns ObjectApiResponse
        if hasattr(resp, "body"):
            return resp.body
        return resp

    def search(self, es_client: Any, index: str, features: Any, *, include_score_breakdown: bool = False, include_explain: bool = False, ):
        groups = self.build_feature_groups(features)
        # FIXED denominator for the entire run: based on initial (strict-intent) groups
        fixed_max_score = self.compute_max_score(groups)
        fixed_max_score = max(1e-6, float(fixed_max_score))
        print("DEBUG fixed_max_score:", fixed_max_score)  # optional



        best_resp: Optional[Dict[str, Any]] = None
        best_q: Optional[Dict[str, Any]] = None
        best_total = -1

        tier_filters: List[Dict[str, Any]] = []
        for step in range(self.max_relax_steps):
            q = self.build_query(groups, include_explain=include_explain, tier_filters=tier_filters,)
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

                display_score = round(self.display_score_for_hit(groups, h, fixed_max_score=fixed_max_score),2)
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


            # Only attach breakdown if requested
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

            # Only keep ES explain if requested
            if not include_explain:
                for h in hits_list:
                    h.pop("_explanation", None)

            # FIX: read totals from resp_dict (not resp)
            hits_obj = resp_dict.get("hits", {})
            total_obj = hits_obj.get("total", 0)
            total = total_obj.get("value", total_obj) if isinstance(total_obj, dict) else total_obj

            if total > best_total:
                best_total, best_resp, best_q = int(total), resp_dict, q

            if total >= self.target_hits:
                return resp_dict, q, groups
            # if total > 0:
            #     return resp_dict, q, groups

            feas = self.feasibility_counts(es_client, index, groups)
            # ----- Task representation relaxation (must forever) -----
            task_fg = next((g for g in groups if g.feature_key == "task"), None)
            if task_fg and task_fg.level == 0:
                # If strict task (canonical/grams only) has 0 matches, enable synonyms (level 1).
                # Task remains MUST because relaxable=False and build_query keeps it in must.
                if feas.get("task", {}).get("strict", 0) == 0:
                    task_fg.level = 1
                    continue
            
            next_fg = self.pick_next_relaxation(groups, feas)
            tier_filters.append(self.build_tier_filter(groups))

            if next_fg is None:
                # Nothing clearly infeasible at current level.
                # Last resort: relax something remaining (strong_prefer then must), feature-by-feature.
                fallback = self._pick_last_resort(groups)
                if fallback is None:
                    break
                fallback.level = min(4, fallback.level + 1)
                continue

            # Relax that feature by one step
            next_fg.level = min(4, next_fg.level + 1)

        # If loop ends, return best attempt (likely 0 hits) for diagnostics
        return best_resp or {"hits": {"total": {"value": 0}, "hits": []}}, best_q or {}, groups

    def _pick_last_resort(self, groups: List[FeatureGroup]) -> Optional[FeatureGroup]:
        """
        When feasibility can't decide, relax remaining relaxables conservatively:
          strong_prefer -> must, lowest weight first, one step.
        """
        cands = [
            fg for fg in groups
            if fg.relaxable
            and fg.priority in ("must", "strong_prefer")
            and fg.level < 4
        ]
        if not cands:
            return None

        def sort_key(fg: FeatureGroup):
            pr_rank = 0 if fg.priority == "strong_prefer" else 1
            return (pr_rank, fg.base_weight, fg.level)

        cands.sort(key=sort_key)
        return cands[0]
