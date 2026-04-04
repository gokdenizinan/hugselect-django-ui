from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any, Tuple, Union

MappingValue = Union[str, List[str]]
MappingType = Dict[str, MappingValue]
from datetime import datetime, timedelta
from EE_Alias_Creation import AliasResolver, make_embedding_provider_factory
# Your existing feature types (import from your codebase)
# from EA_Features import CategoricalFeature, NumericFeature, BoolFeature, RecencyFeature
PreferencePriority = Literal["must", "strong_prefer", "prefer", "avoid"]
NumericOp = Literal["gte", "lte", "gt", "lt", "eq", "approx"]


# ----------------------------
# Plan / stage scaffolding
# ----------------------------

@dataclass
class ClauseSpec:
    """An emitted ES clause + metadata so we can place/relax it later."""
    name: str                 # e.g. "license_name" or "downloads_30d"
    kind: Literal["must", "filter", "should", "must_not"]
    clause: Dict[str, Any]
    relaxable: bool = False   # if True, can move filter->should during relaxation
    alias_level: Literal["canonical", "gram", "synonym"] = "canonical"
    priority: PreferencePriority = "prefer"
    weight: float = 1.0       # used for should-boost (via constant_score)


@dataclass
class RankSignal:
    """A ranking signal (likes, downloads, recency, quality dimension...)."""
    name: str
    signal_type: Literal["field_value_factor", "gauss"]
    field: str
    base_weight: float
    modifier: Optional[str] = None
    missing: float = 0.0
    # gauss params:
    scale: Optional[str] = None
    decay: Optional[float] = None


@dataclass
class Stage:
    """A query stage. Earlier stages are stricter. Later stages loosen constraints."""
    stage_id: int
    clauses: List[ClauseSpec] = field(default_factory=list)
    rank_functions: List[Dict[str, Any]] = field(default_factory=list)
    minimum_should_match: int = 0
    use_task_aliases: bool = False


from EE_Query_Builder_Helper import _extract_score_contributors, _format_no_hits_diagnostics, _build_no_hits_diagnostics_recursive, _extract_bool_query

def _with_scores(resp: Dict[str, Any]) -> Dict[str, Any]:
    hits = resp.get("hits", {}).get("hits", []) or []

    score_breakdown: Dict[str, Dict[str, float]] = {}

    formatted_hits = []
    for h in hits:
        hit_id = h.get("_id")
        formatted_hits.append(
            {
                "id": hit_id,
                "score": h.get("_score"),
                "index": h.get("_index"),
                "source": h.get("_source"),
            }
        )

        expl = h.get("_explanation")
        if hit_id and expl:
            contribs = _extract_score_contributors(expl)

            # If the same feature appears multiple times, sum them.
            agg: Dict[str, float] = {}
            for k, v in contribs:
                agg[k] = agg.get(k, 0.0) + float(v)
            score_breakdown[hit_id] = agg

    return {
        "total": resp.get("hits", {}).get("total", {}),
        "hits": formatted_hits,
        "score_breakdown": score_breakdown,
    }

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
# ----------------------------
# ESQueryBuilder
# ----------------------------
class ESQueryBuilder:
    def __init__(
        self,
        mapping: MappingType,
        *,
        base_signal_weights: Optional[Dict[str, float]] = None,
        priority_multipliers: Optional[Dict[PreferencePriority, float]] = None,
        target_hits: int = 20,
        size: int = 5,
    ):
        self.mapping: MappingType = mapping
        self.resolver = AliasResolver(synonym_provider_factory=make_embedding_provider_factory(min_similarity=0.5), max_synonyms=10,)

        self.target_hits = target_hits
        self.size = size
        self.base_signal_weights = base_signal_weights or {
            "likes": 0.8,
            "downloads_all_time": 0.7,
            "downloads_30d": 1.2,
            "recency": 1.0,
            "quality": 1.5,
            "tag_match": 1.0,
            "functional_match": 1.2,
            "author_match": 10.0,
        }

        self.priority_multipliers = priority_multipliers or {
            "must": 1.6,           # even must constraints can still influence ranking
            "strong_prefer": 1.3,
            "prefer": 1.0,
            "avoid": 0.0,
        }
        self.relax_multiplier = 0.7  # relaxation by synonyms on a category reduces weights

        # Ranking signals (you can add/remove fields here)
        self.rank_signals: List[RankSignal] = [
            RankSignal(
                name="likes",
                signal_type="field_value_factor",
                field=self.mapping.get("likes_field", "Metadata.likes"),
                base_weight=self.base_signal_weights["likes"],
                modifier="log1p",
                missing=0.0,
            ),
            RankSignal(
                name="downloads_all_time",
                signal_type="field_value_factor",
                field=self.mapping.get("downloads_all_time_field", "Metadata.downloads_all_time"),
                base_weight=self.base_signal_weights["downloads_all_time"],
                modifier="log1p",
                missing=0.0,
            ),
            RankSignal(
                name="downloads_30d",
                signal_type="field_value_factor",
                field=self.mapping.get("downloads_30d_field", "Metadata.downloads_last_30_days"),
                base_weight=self.base_signal_weights["downloads_30d"],
                modifier="log1p",
                missing=0.0,
            ),
            # RankSignal(
            #     name="recency",
            #     signal_type="gauss",
            #     field=self.mapping.get("last_modified_field", "Metadata.lastModified"),
            #     base_weight=self.base_signal_weights["recency"],
            #     scale="30d",
            #     decay=0.5,
            # ),
            # If you have an aggregate quality score, prefer that.
            # Otherwise you can add multiple quality dimensions as separate signals.
            RankSignal(
                name="quality",
                signal_type="field_value_factor",
                field=self.mapping.get("functional_suitability_field", "Quality.Functional Suitability.score"),
                base_weight=self.base_signal_weights["quality"],
                modifier=None,
                missing=0.0,
            ),
        ]

    # ----------------------------
    # Priority / weights
    # ----------------------------

    def _fields(self, field_key: str) -> List[str]:
        """Return a list of ES fields for a mapping key (supports str or list[str])."""
        v = self.mapping.get(field_key)
        if v is None:
            raise KeyError(f"Missing mapping for field_key={field_key!r}")
        if isinstance(v, str):
            return [v]
        return list(v)

    def _first_field(self, field_key: str) -> str:
        """Return a single ES field (first) for places that require a single field (range/sort)."""
        return self._fields(field_key)[0]


    def _p(self, priority: Optional[PreferencePriority]) -> PreferencePriority:
        return priority or "prefer"

    def _priority_weight(self, priority: PreferencePriority, base: float) -> float:
        return float(base) * float(self.priority_multipliers.get(priority, 1.0))

    # ----------------------------
    # Clause emitters (pure)
    # ----------------------------

    def _emit_terms(self, field: str, values: List[str]) -> Dict[str, Any]:
        return {"terms": {field: values}}

    def _emit_term(self, field: str, value: Any) -> Dict[str, Any]:
        return {"term": {field: value}}
    
    def _emit_terms_any(self, fields: List[str], values: List[str]) -> Dict[str, Any]:
        return {
            "bool": {
                "should": [{"terms": {f: values}} for f in fields],
                "minimum_should_match": 1
            }
        }
    def _emit_terms_many(self, fields: list[str], values: list[str], k: int = 1) -> dict:
        per_value_groups = [
            {
                "bool": {
                    "should": [{"term": {f: v}} for f in fields],
                    "minimum_should_match": 1,
                }
            }
            for v in values
        ]
        return {
            "bool": {
                "should": per_value_groups,
                "minimum_should_match": int(k),
            }
        }


    def _emit_range(self, field: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return {"range": {field: body}}

    def _wrap_boost(self, clause: Dict[str, Any], weight: float = 1.5) -> Dict[str, Any]:
        """
        Make should-boosts stable and predictable:
        constant_score turns 'matching this clause' into a fixed boost.
        """
        return {"constant_score": {"filter": clause, "boost": float(weight)}}

    def _group_synonyms_per_value(
        self,
        include: List[str],
        syns_vals: List[str],
        syns_weights: List[float],
        *,
        per_value_k: Optional[int] = None,
    ) -> List[List[Tuple[str, float]]]:

        k = int(per_value_k or getattr(self.resolver, "max_synonyms", 10))
        grouped: List[List[Tuple[str, float]]] = []
        idx = 0
        for _ in include:
            block = list(zip(syns_vals[idx: idx + k], syns_weights[idx: idx + k]))
            grouped.append([(s, float(w)) for s, w in block if s and w is not None])
            idx += k
        return grouped


    def _dedupe_keep_max_weight(self, pairs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        best: Dict[str, float] = {}
        for s, w in pairs:
            if s not in best or w > best[s]:
                best[s] = w
        # sort high similarity first
        return sorted(best.items(), key=lambda x: x[1], reverse=True)

        # ----------------------------
    # Feature -> ClauseSpec
    # ----------------------------
    def _categorical_clauses_with_aliases(
        self,
        *,
        resolver: AliasResolver,
        name: str,
        field_key: str,
        pref: Any,  # CategoricalFeature
        force_must: bool = False,
        relaxable_if_strong: bool = True,
        synonym_min_conf: float = 0.35,
        max_synonyms: int = 10,
    ) -> List["ClauseSpec"]:
        if not pref:
            return []

        fields = self._fields(field_key)
        pr = self._p(getattr(pref, "priority", None))

        include = list(getattr(pref, "include", None) or [])
        exclude = list(getattr(pref, "exclude", None) or [])

        out: List["ClauseSpec"] = []
        candidates_sources = [s.partition(".")[2] or s for s in fields]

        # --- Excludes (safe) ---
        if exclude:
            ex_grams: List[str] = resolver.resolve_grams(candidates_sources, exclude)  # <- per your contract
            if ex_grams:
                out.append(
                    ClauseSpec(
                        name=f"{name}.exclude",
                        kind="must_not",
                        clause=self._emit_terms_any(fields, ex_grams),
                        relaxable=False,
                        priority=pr,
                    )
                )

        if not include:
            return out

        # --- Includes ---
        candidates_sources = [s.partition(".")[2] or s for s in fields]
        in_grams: List[str] = resolver.resolve_grams(candidates_sources, include)
        syns_vals, syns_weight = resolver.resolve_syns(candidates_sources, include)  # <- per your contract
        # group back into blocks: [topK for include[0]] + [topK for include[1]] + ...
        k = int(getattr(resolver, "max_synonyms", max_synonyms))
        idx = 0
        pairs: List[Tuple[str, float]] = []
        for _ in include:
            block = list(zip(syns_vals[idx:idx + k], syns_weight[idx:idx + k]))
            pairs.extend([(s, float(w)) for s, w in block if s and w is not None])
            idx += k

        # dedupe synonyms across blocks (keep the best similarity)
        best: Dict[str, float] = {}
        for s, w in pairs:
            if w >= float(synonym_min_conf) and (s not in best or w > best[s]):
                best[s] = w

        syn_pairs = sorted(best.items(), key=lambda x: x[1], reverse=True)[: int(max_synonyms)]

        def synonym_should_clauses(base_weight: float) -> List[Dict[str, Any]]:
            return [
                self._wrap_boost(self._emit_terms_many(fields, [syn]), base_weight * sim)
                for syn, sim in syn_pairs
            ]

        # --- Priority placement ---
        if force_must or pr == "must":
            # must: include itself must-match (across fields)
            out.append(
                ClauseSpec(
                    name=name,
                    kind="must",
                    clause=self._emit_terms_many(fields, include),
                    alias_level="canonical",
                    relaxable=True,
                    priority=pr,
                )
            )

            if in_grams:
                out.append(
                    ClauseSpec(
                        name=f"{name}.gram",
                        kind="must",
                        clause=self._emit_terms_many(fields, in_grams),
                        alias_level="gram",
                        relaxable=True,
                        priority=pr,
                    )
                )

            # IMPORTANT: for must constraints, synonyms should generally be SHOULD boosts, not MUST
            # otherwise you accidentally require a synonym to exist.
            if syn_pairs:
                base = self._priority_weight("must", self.base_signal_weights.get("tag_match", 1.0))
                for i, syn_clause in enumerate(synonym_should_clauses(base)):
                    out.append(
                        ClauseSpec(
                            name=f"{name}.syn.{i}",
                            kind="should",
                            clause=syn_clause,
                            alias_level="synonym",
                            relaxable=True,
                            priority=pr,
                        )
                    )

        elif pr == "strong_prefer":
            # strong_prefer starts as FILTER (so your relaxation loop can move it later)
            out.append(
                ClauseSpec(
                    name=name,
                    kind="must",
                    clause=self._emit_terms_many(fields, include),
                    alias_level="canonical",
                    relaxable=relaxable_if_strong,
                    priority=pr,
                )
            )
            if in_grams:
                out.append(
                    ClauseSpec(
                        name=f"{name}.gram",
                        kind="must",
                        clause=self._emit_terms_many(fields, in_grams),
                        alias_level="gram",
                        relaxable=relaxable_if_strong,
                        priority=pr,
                    )
                )
            if syn_pairs:
                # keep synonyms as SHOULD boosts (even in stage 0)
                base = self._priority_weight(pr, self.base_signal_weights.get("author_match", 10.0)) # tag match , 1 di 
                for i, syn_clause in enumerate(synonym_should_clauses(base)):
                    out.append(
                        ClauseSpec(
                            name=f"{name}.syn.{i}",
                            kind="should",
                            clause=syn_clause,
                            alias_level="synonym",
                            relaxable=True,
                            priority=pr,
                        )
                    )

        elif pr == "prefer":
            base = self._priority_weight(pr, self.base_signal_weights.get("tag_match", 1.0))

            out.append(
                ClauseSpec(
                    name=name,
                    kind="should",
                    clause=self._wrap_boost(self._emit_terms_many(fields, include), base),
                    alias_level="canonical",
                    relaxable=False,
                    priority=pr,
                )
            )
            if in_grams:
                out.append(
                    ClauseSpec(
                        name=f"{name}.gram",
                        kind="should",
                        clause=self._wrap_boost(self._emit_terms_many(fields, in_grams), base),
                        alias_level="gram",
                        relaxable=False,
                        priority=pr,
                    )
                )
            if syn_pairs:
                for i, syn_clause in enumerate(synonym_should_clauses(base)):
                    out.append(
                        ClauseSpec(
                            name=f"{name}.syn.{i}",
                            kind="should",
                            clause=syn_clause,
                            alias_level="synonym",
                            relaxable=False,
                            priority=pr,
                        )
                    )

        elif pr == "avoid":
            # avoid: must_not on grams (safe)
            if in_grams:
                out.append(
                    ClauseSpec(
                        name=name,
                        kind="must_not",
                        clause=self._emit_terms_any(fields, in_grams),
                        relaxable=False,
                        priority=pr,
                    )
                )

        return out

    def _numeric_clauses(
        self,
        *,
        name: str,
        field_key: str,
        pref: Any,  # NumericFeature
        relaxable_if_strong: bool = True,
    ) -> List[ClauseSpec]:
        if not pref:
            return []
        value = getattr(pref, "value", None)
        op = getattr(pref, "op", None)
        if value is None or not op:
            return []

        field = self._first_field(field_key)
        pr = self._p(getattr(pref, "priority", None))

        if op in ("gte", "lte", "gt", "lt"):
            clause = self._emit_range(field, {op: value})
        elif op == "eq":
            clause = self._emit_term(field, value)
        elif op == "approx":
            delta = max(1e-3, float(value) * 0.1)
            clause = self._emit_range(field, {"gte": value - delta, "lte": value + delta})
        else:
            return []

        # Per your spec: strong_pref starts as filter, then can relax.
        if pr == "must":
            return [ClauseSpec(name=name, kind="filter", clause=clause, relaxable=False, priority=pr)]
        if pr == "strong_prefer":
            return [ClauseSpec(name=name, kind="filter", clause=clause, relaxable=relaxable_if_strong, priority=pr)]
        if pr == "prefer":
            w = self._priority_weight(pr, 1.0)
            return [ClauseSpec(name=name, kind="should", clause=self._wrap_boost(clause, w), priority=pr, weight=w)]
        # avoid numeric: you can implement inverse range if you want
        return []

    def _bool_clauses(
        self,
        *,
        name: str,
        field_key: str,
        pref: Any,  # BoolFeature
        relaxable_if_strong: bool = True,
    ) -> List[ClauseSpec]:
        if not pref:
            return []
        val = getattr(pref, "value", None)
        if val is None:
            return []

        field = self._first_field(field_key)
        pr = self._p(getattr(pref, "priority", None))

        clause = self._emit_term(field, str(val))

        if pr == "must":
            return [ClauseSpec(name=name, kind="filter", clause=clause, relaxable=False, priority=pr)]
        if pr == "strong_prefer":
            return [ClauseSpec(name=name, kind="filter", clause=clause, relaxable=relaxable_if_strong, priority=pr)]
        if pr == "prefer":
            w = self._priority_weight(pr, 1.0)
            return [ClauseSpec(name=name, kind="should", clause=self._wrap_boost(clause, w), priority=pr, weight=w)]
        if pr == "avoid":
            return [ClauseSpec(name=name, kind="must_not", clause=clause, relaxable=False, priority=pr)]
        return []

    def _recency_clauses(
        self,
        *,
        name: str,
        date_field_key: str,
        pref: Any,  # RecencyFeature
        now: Optional[datetime] = None,
        relaxable_if_strong: bool = True,
    ) -> List[ClauseSpec]:
        if not pref:
            return []
        max_age_days = getattr(pref, "max_age_days", None)
        if not max_age_days:
            return []

        field = self._first_field(date_field_key)
        pr = self._p(getattr(pref, "priority", None))
        now = now or datetime.utcnow()
        cutoff = now - timedelta(days=int(max_age_days))
        clause = self._emit_range(field, {"gte": cutoff.isoformat()})

        if pr == "must":
            return [ClauseSpec(name=name, kind="filter", clause=clause, relaxable=False, priority=pr)]
        if pr == "strong_prefer":
            return [ClauseSpec(name=name, kind="filter", clause=clause, relaxable=relaxable_if_strong, priority=pr)]
        if pr == "prefer":
            w = self._priority_weight(pr, self.base_signal_weights["recency"])
            return [ClauseSpec(name=name, kind="should", clause=self._wrap_boost(clause, w), priority=pr, weight=w)]
        return []

    # ----------------------------
    # Ranking function builder
    # ----------------------------

    def build_rank_functions(
        self,
        *,
        # user priorities for ranking signals; you can derive these from FeatureBundle if you store them there
        likes_priority: PreferencePriority = "prefer",
        downloads_priority: PreferencePriority = "prefer",
        recency_priority: PreferencePriority = "prefer",
        quality_priority: PreferencePriority = "prefer",
    ) -> List[Dict[str, Any]]:
        functions: List[Dict[str, Any]] = []

        def add(signal: RankSignal, priority: PreferencePriority):
            w = self._priority_weight(priority, signal.base_weight)
            if w <= 0:
                return
            if signal.signal_type == "field_value_factor":
                f = {
                    "field_value_factor": {
                        "field": signal.field,
                        "missing": signal.missing,
                    },
                    "weight": w
                }
                if signal.modifier:
                    f["field_value_factor"]["modifier"] = signal.modifier
                functions.append(f)
            elif signal.signal_type == "gauss":
                functions.append({
                    "gauss": {
                        signal.field: {
                            "origin": "now",
                            "scale": signal.scale or "30d",
                            "decay": signal.decay if signal.decay is not None else 0.5,
                        }
                    },
                    "weight": w
                })

        # Map priorities
        for s in self.rank_signals:
            if s.name == "likes":
                add(s, likes_priority)
            elif s.name in ("downloads_all_time", "downloads_30d"):
                add(s, downloads_priority)
            elif s.name == "recency":
                add(s, recency_priority)
            elif s.name == "quality":
                add(s, quality_priority)
            else:
                add(s, "prefer")

        return functions

    # ----------------------------
    # Build staged plan
    # ----------------------------

    def build_plan(
        self,
        features: Any,  # FeatureBundle
        *,
        noun_phrases: Optional[List[str]] = None,
        inferred_tags: Optional[List[str]] = None,
        inferred_functional: Optional[List[str]] = None,
        task_aliases: Optional[List[str]] = None,
    ) -> List[Stage]:
        """
        Creates stages:
          - Stage 0: strict; strong_pref are filter (relaxable)
          - Stage 1..N: progressively move relaxable filters to should (one group at a time)
          - Fallback stage: use task_aliases if no hits
        """

        all_clauses: List[ClauseSpec] = []
        resolver = self.resolver # replace stub later (!)

        all_clauses += self._categorical_clauses_with_aliases(
            resolver=resolver,
            name="task",
            field_key="task_field",
            pref=features.essential.task,
            force_must=True,           # task always must
            relaxable_if_strong=False,
        )


        # essential domain/author/objective are categorical. strong_pref begins as filter per spec.
        # safece domain ornegi aliasi var suan icin ise yararsa butun categorical clauselara expandle
        # all_clauses += self._categorical_clauses("domain", "domain_field", features.essential.domain)
        all_clauses += self._categorical_clauses_with_aliases(resolver=resolver, name="domain", field_key="domain_field", pref=features.essential.domain) 
        all_clauses += self._categorical_clauses_with_aliases(resolver=resolver, name="author", field_key="author_field", pref=features.essential.author)
        all_clauses += self._categorical_clauses_with_aliases(resolver=resolver, name="objective", field_key="objective_field", pref=features.essential.objective)

        # --- Preferences ---
        all_clauses += self._categorical_clauses_with_aliases(resolver=resolver, name="license_name", field_key="license_field", pref=features.preferences.license_name)
        all_clauses += self._categorical_clauses_with_aliases(resolver=resolver, name="library_name", field_key="library_name_field", pref=features.preferences.library_name)
        all_clauses += self._categorical_clauses_with_aliases(resolver=resolver, name="basemodels", field_key="basemodels_field", pref=features.preferences.basemodels)
        all_clauses += self._categorical_clauses_with_aliases(resolver=resolver, name="datasets", field_key="datasets_field", pref=features.preferences.datasets)
        all_clauses += self._categorical_clauses_with_aliases(resolver=resolver, name="language", field_key="language_field", pref=features.preferences.language)
        all_clauses += self._categorical_clauses_with_aliases(resolver=resolver, name="metrics", field_key="metrics_field", pref=features.preferences.metrics)

        all_clauses += self._numeric_clauses(name = "downloads_all_time", field_key="downloads_all_time_field", pref = features.preferences.downloads_all_time)
        all_clauses += self._numeric_clauses(name ="downloads_30d", field_key="downloads_30d_field",pref = features.preferences.downloads_last_30_days)
        all_clauses += self._numeric_clauses(name ="likes", field_key="likes_field", pref =features.preferences.likes)
        all_clauses += self._numeric_clauses(name ="file_count", field_key="file_count_field",pref = features.preferences.file_count)
        all_clauses += self._numeric_clauses(name ="tensors_total", field_key="tensors_total_field",pref = features.preferences.tensors_total)
        all_clauses += self._numeric_clauses(name ="usedStorage",field_key= "used_storage_field", pref =features.preferences.usedStorage)

        all_clauses += self._bool_clauses(name ="gated",field_key= "gated_field", pref =features.preferences.gated)

        all_clauses += self._recency_clauses(name= "lastModified", date_field_key="last_modified_field", pref=features.preferences.lastModified)

        # --- Ranking functions (derive priorities however you like) ---
        rank_functions = self.build_rank_functions(
            likes_priority=getattr(features.preferences.likes, "priority", "prefer") if getattr(features.preferences, "likes", None) else "prefer",
            downloads_priority="prefer",
            recency_priority=getattr(features.preferences.lastModified, "priority", "prefer") if getattr(features.preferences, "lastModified", None) else "prefer",
            quality_priority="prefer",
        )

        # --- Stage 0 ---
        stages: List[Stage] = [Stage(stage_id=0, clauses=list(all_clauses), rank_functions=rank_functions)]

        # Build relaxation stages by moving relaxable Must clauses -> SHOULD (with boost)
        # relaxables = [c for c in all_clauses if c.kind == "must" and c.relaxable]
        relaxables = [c for c in all_clauses if c.relaxable and c.kind in ("must", "filter")]


        # Move one relaxable at a time (you can change grouping logic to “least important first”)
        for i, clause_to_relax in enumerate(relaxables, start=1):
            prev = stages[-1]

            new_clauses: List[ClauseSpec] = []
            for c in prev.clauses:
                if c is clause_to_relax:
                    # Move filter -> should with a boost derived from its priority
                    pr = c.priority
                    base = self.base_signal_weights.get("tag_match", 1.0)
                    w = self._priority_weight(pr, base)
                    new_clauses.append(ClauseSpec(
                        name=c.name,
                        kind="should",
                        clause=self._wrap_boost(c.clause, w),
                        relaxable=False,
                        priority=pr,
                        weight=w,
                    ))
                else:
                    new_clauses.append(c)

            stages.append(Stage(stage_id=i, clauses=new_clauses, rank_functions=rank_functions))

        # Fallback stage using task_aliases if provided
        if task_aliases:
            # Copy last stage and replace task clause with alias terms in must
            last = stages[-1]
            replaced: List[ClauseSpec] = []
            for c in last.clauses:
                if c.name == "task" and c.kind == "must":
                    task_fields = self._fields("task_field")
                    replaced.append(ClauseSpec(
                        name="task",
                        kind="must",
                        clause=self._emit_terms_many(task_fields, task_aliases),
                        relaxable=False,
                        priority="must",
                    ))
                else:
                    replaced.append(c)
            stages.append(Stage(stage_id=len(stages), clauses=replaced, rank_functions=rank_functions, use_task_aliases=True))

        return stages

    # ----------------------------
    # Stage -> ES query
    # ----------------------------

    def build_stage_query(self, stage: Stage) -> Dict[str, Any]:
        must, flt, should, must_not = [], [], [], []

        for c in stage.clauses:
            if c.kind == "must":
                must.append(c.clause)
            elif c.kind == "filter":
                flt.append(c.clause)
            elif c.kind == "should":
                should.append(c.clause)
            elif c.kind == "must_not":
                must_not.append(c.clause)

        bool_query: Dict[str, Any] = {"bool": {}}
        if must: bool_query["bool"]["must"] = must
        if flt: bool_query["bool"]["filter"] = flt
        if should: bool_query["bool"]["should"] = should
        if must_not: bool_query["bool"]["must_not"] = must_not

        # Keep should optional; you can change this policy
        bool_query["bool"]["minimum_should_match"] = stage.minimum_should_match

        query: Dict[str, Any] = {
            "explain": True,
            "query": {
                "function_score": {
                    "query": bool_query,
                    "score_mode": "sum",
                    "boost_mode": "sum",
                    "functions": stage.rank_functions,
                }
            },
            "size": self.size,
            "sort": [
                {"_score": {"order": "desc"}},
                {self.mapping.get("last_modified_field", "Metadata.lastModified"): {"order": "desc"}},
                {self.mapping.get("downloads_30d_field", "Metadata.downloads_last_30_days"): {"order": "desc"}},
            ],
        }
        return query

    # ----------------------------
    # Optional: execute relaxation loop (needs ES client)
    # ----------------------------

    def search(self, es_client: Any, index: str, stages: List[Stage]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        best_resp: Optional[Dict[str, Any]] = None
        best_query: Optional[Dict[str, Any]] = None
        best_hits = -1

        # 1) Try each stage and keep the best response/query
        for stage in stages:
            q = self.build_stage_query(stage)
            resp = es_client.search(index=index, body=q)

            hits_total = resp.get("hits", {}).get("total", 0)
            total = hits_total.get("value", hits_total) if isinstance(hits_total, dict) else hits_total

            if total > best_hits:
                best_hits, best_resp, best_query = total, resp, q

            # early exit if we met the target
            if total >= self.target_hits:
                return _with_scores(resp), q

        # 2) If we never ran anything (edge case)
        if best_resp is None or best_query is None:
            return {"hits": [], "note": "No stages executed / no response received."}, {}

        # 3) If the best stage has hits, return it normally
        hits = best_resp.get("hits", {}).get("hits", []) or []
        if hits:
            return _with_scores(best_resp), best_query

        # 4) Zero-hits diagnostics (based on best_query)
        bool_q = _extract_bool_query(best_query)
        if not bool_q:
            return {
                "hits": [],
                "note": "No hits and could not extract bool query for diagnostics."
            }, best_query

        # Use recursive diagnostics if you implemented it; otherwise _build_no_hits_diagnostics
        diag_body = _build_no_hits_diagnostics_recursive(bool_q)  # or _build_no_hits_diagnostics(bool_q)
        diag_resp = es_client.search(index=index, body=diag_body)
        diagnosis = _format_no_hits_diagnostics(diag_resp)

        return {
            "hits": [],
            "no_hits_diagnosis": diagnosis
        }, best_query


