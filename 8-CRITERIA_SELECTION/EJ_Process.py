from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any, Tuple
from datetime import datetime, timedelta
from EE_Alias_Creation import AliasResolver, SynonymAlias, stub_synonym_provider

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

def _expand_values_with_aliases(
    resolver: AliasResolver,
    feature_name: str,
    values: List[str],
) -> Tuple[List[str], List[SynonymAlias]]:
    all_grams: List[str] = []
    all_syns: List[SynonymAlias] = []

    seen_grams = set()
    seen_syns = set()

    for v in values:
        grams, syns = resolver.resolve(feature_name, v)

        for g in grams:
            k = g.lower()
            if k not in seen_grams:
                seen_grams.add(k)
                all_grams.append(g)

        for s in syns:
            k = s.value.lower()
            if k not in seen_syns:
                seen_syns.add(k)
                all_syns.append(s)

    return all_grams, all_syns


def _with_scores(resp: Dict[str, Any]) -> Dict[str, Any]:
    hits = resp.get("hits", {}).get("hits", []) or []
    return {
        "total": resp.get("hits", {}).get("total", {}),
        "hits": [
            {
                "id": h.get("_id"),
                "score": h.get("_score"),
                "index": h.get("_index"),
                "source": h.get("_source"),
            }
            for h in hits
        ],
    }


# ----------------------------
# ESQueryBuilder
# ----------------------------

class ESQueryBuilder:
    """
    Builder that:
      - emits clauses from FeatureBundle
      - creates staged queries with progressive relaxation
      - ranks using function_score (likes/downloads/recency/quality + match boosts)
    """

    def __init__(
        self,
        mapping: Dict[str, str],
        *,
        # Inherent/base importance of ranking signals (system priors)
        base_signal_weights: Optional[Dict[str, float]] = None,
        # How much the user's priority multiplies the base weights
        priority_multipliers: Optional[Dict[PreferencePriority, float]] = None,
        # How many results you want
        target_hits: int = 20,
        size: int = 5,
    ):
        self.mapping = mapping
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
        }

        self.priority_multipliers = priority_multipliers or {
            "must": 1.6,           # even must constraints can still influence ranking
            "strong_prefer": 1.3,
            "prefer": 1.0,
            "avoid": 0.0,
        }

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

    def _emit_range(self, field: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return {"range": {field: body}}

    def _wrap_boost(self, clause: Dict[str, Any], weight: float) -> Dict[str, Any]:
        """
        Make should-boosts stable and predictable:
        constant_score turns 'matching this clause' into a fixed boost.
        """
        return {"constant_score": {"filter": clause, "boost": float(weight)}}

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
        # How strong synonym matches are compared to canonical
        synonym_boost_factor: float = 0.6,
        # Minimum confidence to use synonym at all
        synonym_min_conf: float = 0.35,
    ) -> List["ClauseSpec"]:
        if not pref:
            return []

        field = self.mapping[field_key]
        pr = self._p(getattr(pref, "priority", None))

        include = list(getattr(pref, "include", None) or [])
        exclude = list(getattr(pref, "exclude", None) or [])

        out: List["ClauseSpec"] = []

        # Excludes: expand with grammatical aliases only (safe)
        if exclude:
            ex_grams, _ = _expand_values_with_aliases(resolver, name, exclude)
            out.append(ClauseSpec(
                name=f"{name}.exclude",
                kind="must_not",
                clause={"terms": {field: ex_grams}},
                relaxable=False,
                priority=pr,
            ))

        if not include:
            return out

        # Includes: canonical + grammatical go into same bucket (must/filter/should)
        in_grams, in_syns = _expand_values_with_aliases(resolver, name, include)

        # ---- place grammatical (safe) terms according to priority rules ----
        if force_must or pr == "must":
            out.append(ClauseSpec(
                name=name,
                kind="must",
                clause={"terms": {field: in_grams}},
                relaxable=False,
                priority=pr,
            ))
        elif pr == "strong_prefer":
            out.append(ClauseSpec(
                name=name,
                kind="filter",
                clause={"terms": {field: in_grams}},
                relaxable=relaxable_if_strong,
                priority=pr,
            ))
        elif pr == "prefer":
            w = self._priority_weight(pr, self.base_signal_weights.get("tag_match", 1.0))
            out.append(ClauseSpec(
                name=name,
                kind="should",
                clause=self._wrap_boost({"terms": {field: in_grams}}, w),
                relaxable=False,
                priority=pr,
                weight=w,
            ))
        elif pr == "avoid":
            out.append(ClauseSpec(
                name=name,
                kind="must_not",
                clause={"terms": {field: in_grams}},
                relaxable=False,
                priority=pr,
            ))

        # ---- synonym aliases: SHOULD boosts only (by default) ----
        # even if priority is must/strong_prefer, we still keep synonyms as soft
        if in_syns and pr != "avoid":
            # Base for this type of match; you can use per-feature base weights if you want.
            base = self.base_signal_weights.get("tag_match", 1.0)
            # Make synonyms weaker than canonical
            base *= synonym_boost_factor

            for syn in in_syns:
                if syn.confidence < synonym_min_conf:
                    continue
                # confidence influences boost
                w = base * float(syn.confidence)

                out.append(ClauseSpec(
                    name=f"{name}.synonym",
                    kind="should",
                    clause=self._wrap_boost({"terms": {field: [syn.value]}}, w),
                    relaxable=False,
                    priority="prefer",
                    weight=w,
                ))

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

        field = self.mapping[field_key]
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

        field = self.mapping[field_key]
        pr = self._p(getattr(pref, "priority", None))

        clause = self._emit_term(field, str(bool(val)).lower())

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

        field = self.mapping[date_field_key]
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
    # Special matching (noun phrases, inferred tags, functional features)
    # ----------------------------

    def add_special_matches( # emin degilim bu lazim mi?, asil amac functional featurelarla taglerin bir suru kategoriyle matchlenmeye calismasi, ayni zamanda terseten olarak domain ile onjective de bir suru seyle matchlenebilir, hatta task bile( ama burdami yapilmali o?)
        self,
        clauses: List[ClauseSpec],
        *,
        noun_phrases: Optional[List[str]] = None,
        inferred_tags: Optional[List[str]] = None,
        inferred_functional: Optional[List[str]] = None,
        weight_scale: float = 0.7,
    ) -> None:
        """
        All special matches are SHOULD boosts (not filters), unless you later mark them must explicitly.
        You can swap 'terms' for 'match' if your fields are analyzed; here we assume tags/functional are keywords.
        """
        noun_phrases = noun_phrases or []
        inferred_tags = inferred_tags or []
        inferred_functional = inferred_functional or []

        tags_field = self.mapping.get("tags_field", "Metadata.tags")
        functional_field = self.mapping.get("functional_field", "Features")  # <-- adjust to your mapping key

        if inferred_tags:
            w = self.base_signal_weights["tag_match"] * weight_scale
            clauses.append(ClauseSpec(
                name="special.inferred_tags",
                kind="should",
                clause=self._wrap_boost(self._emit_terms(tags_field, inferred_tags), w),
                relaxable=False,
                priority="prefer",
                weight=w,
            ))

        if inferred_functional:
            w = self.base_signal_weights["functional_match"] * weight_scale
            clauses.append(ClauseSpec(
                name="special.inferred_functional",
                kind="should",
                clause=self._wrap_boost(self._emit_terms(functional_field, inferred_functional), w),
                relaxable=False,
                priority="prefer",
                weight=w,
            ))

        if noun_phrases:
            # If you store noun phrases into tags/Features during indexing, re-use them here.
            # Otherwise, consider a multi_match on text fields. Keep it as soft boost.
            w = 0.6 * weight_scale
            clauses.append(ClauseSpec(
                name="special.noun_phrases",
                kind="should",
                clause=self._wrap_boost(self._emit_terms(tags_field, noun_phrases), w),
                relaxable=False,
                priority="prefer",
                weight=w,
            ))

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
    # (!) ilk once essential preference, etc classlarinin her documanda dogru oldugunu kontrol et (model_id yok mesela bagzilarinda)
    # (!) qualityi filterdan tamamen cikar
    # (!) field key aranilan lokasyon ise -> bir kac tane eklenebilir taski birden fazla yerde arayan falan
    # (!) ancak bunlarin arasina or gibi birsey konmali ki 
    # (!) eklenmesi gereken featurelar var hala
    # (!) stagelerde hangi filtrenin ilk should gececegi sirasini ayarla
    # (!) task aliases olayni ayarla guzelce -> syns alias ile baglanti sirasini ayarla

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
        resolver = AliasResolver(synonym_provider=stub_synonym_provider)  # replace stub later (!)


        # --- MUST: task always must (plus "must" features) ---
        # all_clauses += self._categorical_clauses(
        #     name="task",
        #     field_key="task_field",
        #     pref=features.essential.task,
        #     force_must=True,
        #     relaxable_if_strong=False,
        # )
        # taskin aliasli hali
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

        all_clauses += self._numeric_clauses(name = "downloads_all_time", field_key="downloads_all_time_field", pref = features.preferences.downloads_all_time)
        all_clauses += self._numeric_clauses(name ="downloads_30d", field_key="downloads_30d_field",pref = features.preferences.downloads_last_30_days)
        all_clauses += self._numeric_clauses(name ="likes", field_key="likes_field", pref =features.preferences.likes)
        all_clauses += self._numeric_clauses(name ="file_count", field_key="file_count_field",pref = features.preferences.file_count)
        all_clauses += self._numeric_clauses(name ="tensors_total", field_key="tensors_total_field",pref = features.preferences.tensors_total)
        all_clauses += self._numeric_clauses(name ="usedStorage",field_key= "used_storage_field", pref =features.preferences.usedStorage)

        all_clauses += self._bool_clauses(name ="gated",field_key= "gated_field", pref =features.preferences.gated)

        all_clauses += self._recency_clauses(name= "lastModified", date_field_key="last_modified_field", pref=features.preferences.lastModified)

        # --- Quality: treat user_val>=4 as must/filter by emitting filter clauses or should boosts.
        # You can keep your existing _add_quality logic; here’s a simplified version:
        quality = getattr(features, "quality", None)
        if quality:
            quality_map = {
                "Functional_Suitability": "functional_suitability_field",
                "Compatibility": "compatibility_field",
                "Performance_Efficiency": "performance_efficiency_field",
                "Reliability": "reliability_field",
                "Interaction_Capability": "interaction_capability_field",
                "Security": "security_field",
                "Maintainability": "maintainability_field",
                "Flexibility": "flexibility_field",
            }
            for attr, field_key in quality_map.items():
                user_val = getattr(quality, attr, None)
                if user_val is None or user_val <= 0:
                    continue

                # If your scores are 0..1 in ES and user_val is 1..5:
                normalized = (float(user_val) - 1.0) / 4.0

                clause = self._emit_range(self.mapping[field_key], {"gte": normalized})

                if user_val >= 4:
                    all_clauses.append(ClauseSpec(
                        name=f"quality.{attr}",
                        kind="filter",          # strong quality starts strict
                        clause=clause,
                        relaxable=True,         # allow relaxation later if needed
                        priority="strong_prefer",
                    ))
                else:
                    w = self._priority_weight("prefer", self.base_signal_weights["quality"])
                    all_clauses.append(ClauseSpec(
                        name=f"quality.{attr}",
                        kind="should",
                        clause=self._wrap_boost(clause, w),
                        relaxable=False,
                        priority="prefer",
                        weight=w,
                    ))

        # --- Special matches as SHOULD boosts ---
        self.add_special_matches(
            all_clauses,
            noun_phrases=noun_phrases,
            inferred_tags=inferred_tags,
            inferred_functional=inferred_functional,
            weight_scale=0.7,
        )

        # --- Ranking functions (derive priorities however you like) ---
        rank_functions = self.build_rank_functions(
            likes_priority=getattr(features.preferences.likes, "priority", "prefer") if getattr(features.preferences, "likes", None) else "prefer",
            downloads_priority="prefer",
            recency_priority=getattr(features.preferences.lastModified, "priority", "prefer") if getattr(features.preferences, "lastModified", None) else "prefer",
            quality_priority="prefer",
        )

        # --- Stage 0 ---
        stages: List[Stage] = [Stage(stage_id=0, clauses=list(all_clauses), rank_functions=rank_functions)]

        # Build relaxation stages by moving relaxable FILTER clauses -> SHOULD (with boost)
        relaxables = [c for c in all_clauses if c.kind == "filter" and c.relaxable]

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
                    task_field = self.mapping["task_field"]
                    replaced.append(ClauseSpec(
                        name="task",
                        kind="must",
                        clause=self._emit_terms(task_field, task_aliases),
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
        best_resp = None
        best_query = None
        best_hits = -1

        for stage in stages:
            q = self.build_stage_query(stage)
            resp = es_client.search(index=index, body=q)

            hits_total = resp.get("hits", {}).get("total", {})
            total = hits_total.get("value", hits_total) if isinstance(hits_total, dict) else hits_total

            if total > best_hits:
                best_hits, best_resp, best_query = total, resp, q

            if total >= self.target_hits:
                return _with_scores(resp), q

        return _with_scores(best_resp or {}), (best_query or {})

