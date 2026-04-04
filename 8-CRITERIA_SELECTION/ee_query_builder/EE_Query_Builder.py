from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from EA_Features import (PreferenceFeatures, CategoricalFeature, NumericFeature, BoolFeature, RecencyFeature,
                          PreferencePriority, NumericOp)



query_mapping = {
    # high-level intent
    "task_field": "Metadata.pipeline_tag",   # e.g. "text-classification"
    "domain_field": "Metadata.tags",         # domain as tags (e.g. "medical", "finance")
    "author_field": "author",
    "objective_field": "Metadata.tags",      # you can reserve special tags for objectives

    # metadata / constraints
    "license_field": "Metadata.license",
    "downloads_all_time_field": "Metadata.downloads_all_time",
    "downloads_30d_field": "Metadata.downloads_last_30_days",
    "file_count_field": "Metadata.file_count",
    "gated_field": "Metadata.gated",                 # keyword: "true"/"false" or similar
    "private_field": "Metadata.private",             # keyword: "true"/"false"
    "library_name_field": "Metadata.library_name",
    "model_type_field": "Metadata.model_type",
    "tags_field": "Metadata.tags",
    "basemodels_field": "Metadata.basemodels",
    "datasets_field": "Metadata.datasets",
    "tensors_total_field": "Metadata.tensors_total",
    "used_storage_field": "Metadata.usedStorage",
    "last_modified_field": "Metadata.lastModified",  # date

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

from elasticsearch import Elasticsearch, helpers
from D_elastic_2 import load_mapping, recreate_index, bulk_index


ES_URL = "http://localhost:9200"  # change if needed
INDEX_NAME = "models_02"             # name of the ES index
MAPPING_FILE = "8-CRITERIA_SELECTION/es_mapping_T9.json"
DATA_FOLDER = "HF-Models-T9"

DEFAULT_WEIGHTS = {
    # Universal preference strengths
    "strong_prefer": 5.0,
    "prefer": 2.0,

    # Optional special cases by feature
    "domain_strong_prefer": 4.0,
    "domain_prefer": 1.5,

    "author_strong_prefer": 3.0,
    "author_prefer": 1.2,
}

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Literal
from datetime import datetime, timedelta
from EA_Features import CategoricalFeature, NumericFeature, BoolFeature, RecencyFeature

PreferencePriority = Literal["must", "strong_prefer", "prefer", "avoid"]
NumericOp = Literal["gte", "lte", "gt", "lt", "eq", "approx"]

class ESQueryBuilder:
    def __init__(self, mapping: Dict, weights: Dict):
        self.mapping = mapping
        self.weights = weights

    # ---------- helpers ----------

    def _priority_or_default(self, priority: Optional[PreferencePriority]) -> PreferencePriority:
        return priority or "prefer"

    def _boost_for(self, feature_name: str, priority: PreferencePriority) -> float:
        key = f"{feature_name}_{priority}_boost"
        return float(self.weights.get(key, 1.0))

    def _add_categorical(
        self,
        pref: Optional[CategoricalFeature],
        feature_name: str,
        field_key: str,
        must_clauses: List[Dict],
        should_clauses: List[Dict],
        must_not_clauses: List[Dict],
        force_must: bool = False,
    ):
        if not pref:
            return

        priority = self._priority_or_default(pref.priority)
        field = self.mapping[field_key]

        # excludes → must_not
        if pref.exclude:
            must_not_clauses.append({"terms": {field: pref.exclude}})

        if not pref.include:
            return

        if force_must or priority == "must":
            must_clauses.append({"terms": {field: pref.include}})
        elif priority in ("strong_prefer", "prefer"):
            should_clauses.append({"terms": {field: pref.include}})
        elif priority == "avoid":
            must_not_clauses.append({"terms": {field: pref.include}})

    def _add_numeric(
        self,
        pref: Optional[NumericFeature],
        feature_name: str,
        field_key: str,
        filter_clauses: List[Dict],
        should_clauses: List[Dict],
    ):
        if not pref or pref.value is None or not pref.op:
            return

        priority = self._priority_or_default(pref.priority)
        field = self.mapping[field_key]
        value = pref.value

        if pref.op in ("gte", "lte", "gt", "lt"):
            query = {"range": {field: {pref.op: value}}}
        elif pref.op == "eq":
            query = {"term": {field: value}}
        elif pref.op == "approx":
            delta = max(1e-3, value * 0.1)
            query = {"range": {field: {"gte": value - delta, "lte": value + delta}}}
        else:
            return

        if priority == "must":
            filter_clauses.append(query)
        elif priority in ("strong_prefer", "prefer"):
            should_clauses.append(query)
        elif priority == "avoid":
            # optional: add inverse constraint in must_not
            pass

    def _add_bool(
        self,
        pref: Optional[BoolFeature],
        feature_name: str,
        field_key: str,
        filter_clauses: List[Dict],
        should_clauses: List[Dict],
        must_not_clauses: List[Dict],
    ):
        if not pref or pref.value is None:
            return

        priority = self._priority_or_default(pref.priority)
        field = self.mapping[field_key]
        desired = pref.value

        # NOTE: your mapping uses keyword, so you'd store "true"/"false" or similar.
        term_query = {"term": {field: str(desired).lower()}}

        if priority == "must":
            filter_clauses.append(term_query)
        elif priority in ("strong_prefer", "prefer"):
            should_clauses.append(term_query)
        elif priority == "avoid":
            must_not_clauses.append(term_query)

    def _add_recency(
        self,
        pref: Optional[RecencyFeature],
        feature_name: str,
        date_field_key: str,
        filter_clauses: List[Dict],
        should_clauses: List[Dict],
        now: Optional[datetime] = None,
    ):
        if not pref or not pref.max_age_days:
            return

        priority = self._priority_or_default(pref.priority)
        field = self.mapping[date_field_key]
        now = now or datetime.utcnow()
        cutoff = now - timedelta(days=pref.max_age_days)

        range_query = {"range": {field: {"gte": cutoff.isoformat()}}}

        if priority == "must":
            filter_clauses.append(range_query)
        elif priority in ("strong_prefer", "prefer"):
            should_clauses.append(range_query)
    
    def _add_quality(
        self,
        quality: "QualityFeatures",
        mapping: Dict,
        must_clauses: List[Dict],
        should_clauses: List[Dict],
    ):

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

        for attr_name, field_key in quality_map.items():
            user_val = getattr(quality, attr_name)

            # Skip missing values
            if user_val is None or user_val <= 0:
                continue

            # Normalization 1–5 → 0.0–1.0
            normalized = (user_val - 1) / 10.0 # 4 e bolunuyor orjinalinde

            es_field = mapping[field_key]

            # priority decision
            if user_val >= 4:          # strong importance
                priority = "must"
            elif user_val == 3:
                priority = "strong_prefer"
            else:  # 1 or 2
                priority = "prefer"

            range_clause = {
                "range": {
                    es_field: {
                        "gte": normalized
                    }
                }
            }

            if priority == "must":
                must_clauses.append(range_clause)
            else:
                should_clauses.append(range_clause)


    # ---------- main build ----------

    def build(self, features: "FeatureBundle") -> Dict:
        must_clauses: List[Dict] = []
        filter_clauses: List[Dict] = []
        should_clauses: List[Dict] = []
        must_not_clauses: List[Dict] = []

        # === INTENT: task, domain, author, objective ===

        # # task: main pipeline_tag → force must
        # task_pref = features.essential.task

        # normalized_pref = CategoricalFeature(
        #     include=[s.replace(" ", "-") for s in task_pref.include],
        #     exclude=[s.replace(" ", "-") for s in task_pref.exclude],
        #     bucket=task_pref.bucket,
        #     priority=task_pref.priority,
        # )
        self._add_categorical(
            # pref=normalized_pref,              # CategoricalFeature
            pref = features.essential.task,
            feature_name="task",
            field_key="task_field",                 # "Metadata.pipeline_tag"
            must_clauses=must_clauses,
            should_clauses=should_clauses,
            must_not_clauses=must_not_clauses,
            force_must=True,
        )

        # domain modeled as tags
        self._add_categorical(
            pref=features.essential.domain,            # CategoricalFeature
            feature_name="domain",
            field_key="domain_field",               # "Metadata.tags"
            must_clauses=must_clauses,
            should_clauses=should_clauses,
            must_not_clauses=must_not_clauses,
        )

        self._add_categorical(
            pref=features.essential.author,            # CategoricalFeature
            feature_name="author",
            field_key="author_field",               # "author"
            must_clauses=must_clauses,
            should_clauses=should_clauses,
            must_not_clauses=must_not_clauses,
        )

        self._add_categorical(
            pref=features.essential.objective,         # CategoricalFeature
            feature_name="objective",
            field_key="objective_field",            # "Metadata.tags" with special tags like "objective:quality"
            must_clauses=must_clauses,
            should_clauses=should_clauses,
            must_not_clauses=must_not_clauses,
        )

        # === NUMERIC: metadata (downloads, params, etc.) ===

        self._add_numeric(
            pref=features.preferences.downloads_all_time,   # NumericFeature
            feature_name="downloads_all_time",
            field_key="downloads_all_time_field",           # "Metadata.downloads_all_time"
            filter_clauses=filter_clauses,
            should_clauses=should_clauses,
        )

        self._add_numeric(
            pref=features.preferences.downloads_last_30_days,        # NumericFeature
            feature_name="downloads_last_30_days",
            field_key="downloads_30d_field",                # "Metadata.downloads_last_30_days"
            filter_clauses=filter_clauses,
            should_clauses=should_clauses,
        )

        self._add_numeric(
            pref=features.preferences.tensors_total,         # NumericFeature (for tensors_total)
            feature_name="tensors_total",
            field_key="tensors_total_field",                # "Metadata.tensors_total"
            filter_clauses=filter_clauses,
            should_clauses=should_clauses,
        )

        self._add_numeric(
            pref=features.preferences.file_count,         # NumericFeature (for tensors_total)
            feature_name="file_count",
            field_key="file_count_field",                # "Metadata.tensors_total"
            filter_clauses=filter_clauses,
            should_clauses=should_clauses,
        )

        self._add_numeric(
            pref=features.preferences.likes,         # NumericFeature (for tensors_total)
            feature_name="likes",
            field_key="likes_field",                # "Metadata.tensors_total"
            filter_clauses=filter_clauses,
            should_clauses=should_clauses,
        )

        self._add_numeric(
            pref=features.preferences.usedStorage,         # NumericFeature (for tensors_total)
            feature_name="usedStorage",
            field_key="usedStorage_field",                # "Metadata.tensors_total"
            filter_clauses=filter_clauses,
            should_clauses=should_clauses,
        )

        # === NUMERIC: Quality.*.score ===
        self._add_quality(
            quality=features.quality,          # QualityFeatures dataclass
            mapping=self.mapping,
            must_clauses=must_clauses,
            should_clauses=should_clauses,
        )
        # Functional Feature ? ekle

        # == Other Cathgorical: 
        self._add_categorical(
            pref=features.preferences.basemodels,         # CategoricalFeature
            feature_name="basemodels",
            field_key="basemodels_field",           
            must_clauses=must_clauses,
            should_clauses=should_clauses,
            must_not_clauses=must_not_clauses,
        )
        self._add_categorical(
            pref=features.preferences.license_name,         # CategoricalFeature
            feature_name="license_name",
            field_key="license_name_field",           
            must_clauses=must_clauses,
            should_clauses=should_clauses,
            must_not_clauses=must_not_clauses,
        )
        self._add_categorical(
            pref=features.preferences.library_name,         # CategoricalFeature
            feature_name="library_name",
            field_key="library_name_field",           
            must_clauses=must_clauses,
            should_clauses=should_clauses,
            must_not_clauses=must_not_clauses,
        )
        self._add_categorical(
            pref=features.preferences.datasets,         # CategoricalFeature
            feature_name="datasets",
            field_key="datasets_field",           
            must_clauses=must_clauses,
            should_clauses=should_clauses,
            must_not_clauses=must_not_clauses,
        )

        # === BOOL / keyword flags: gated, private, commercial use etc. ===

        self._add_bool(
            pref=features.preferences.gated,                 # BoolFeature
            feature_name="gated",
            field_key="gated_field",                   # "Metadata.gated"
            filter_clauses=filter_clauses,
            should_clauses=should_clauses,
            must_not_clauses=must_not_clauses,
        )
        self._add_bool(
            pref=features.preferences.private,               # BoolFeature
            feature_name="private",
            field_key="private_field",                 # "Metadata.private"
            filter_clauses=filter_clauses,
            should_clauses=should_clauses,
            must_not_clauses=must_not_clauses,
        )
        self._add_bool(
            pref=features.preferences.spaces,               # BoolFeature
            feature_name="spaces",
            field_key="spaces_field",                 # "Metadata.private"
            filter_clauses=filter_clauses,
            should_clauses=should_clauses,
            must_not_clauses=must_not_clauses,
        )

        # === RECENCY: lastModified ===

        self._add_recency(
            pref=features.preferences.lastModified,                     # RecencyFeature
            feature_name="recency",
            date_field_key="last_modified_field",      # "Metadata.lastModified"
            filter_clauses=filter_clauses,
            should_clauses=should_clauses,
        )

        query = {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses,
                "should": should_clauses,
                "must_not": must_not_clauses,
                "minimum_should_match": 0 if not should_clauses else 1,
            }
        }

        return {
            "query": query,
            "size": 20,
        }

# BUNU CALISTIRMAYA CALIS
def example_query(es: Elasticsearch, index_name: str, body) -> None:

    res = es.search(index=index_name, body=body)

    print("\nModels Matching Query:")
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        model_id = src.get("modelID")
        tags = src.get("tags")
        features = src.get("Features")
        pipeline_tag = src.get("Metadata", {}).get("pipeline_tag")
        print(f"- modelID={model_id!r}, tags={tags!r}, Features={features!r}, pipeline_tag={pipeline_tag!r}")



if __name__ == "__main__":
    
    # body_2 = {'query': {'bool': {'must': [{'terms': {'Metadata.pipeline_tag': ['image-classification']}}, {'range': {'Quality.Functional Suitability.score': {'gte': 0.4}}}], 'filter': [], 'should': [], 'must_not': [], 'minimum_should_match': 0}}, 'size': 20}

    # body_3 = {'query': {'bool': {'must': [{'terms': {'Metadata.pipeline_tag': ['text-generation']}}, {'terms': {'Metadata.tags': ['permissive licenses', 'MIT', 'Apache-2', 'open', 'widely usable']}}, {'range': {'Quality.Functional Suitability.score': {'gte': 0.4}}}, {'range': {'Quality.Compatibility.score': {'gte': 0.3}}}, {'range': {'Quality.Flexibility.score': {'gte': 0.4}}}], 'filter': [], 'should': [{'terms': {'author': ['Meta', 'closely related authors']}}, {'term': {'Metadata.gated': 'false'}}, {'term': {'Metadata.private': 'false'}}], 'must_not': [{'terms': {'Metadata.tags': ['gated', 'restricted']}}], 'minimum_should_match': 1}}, 'size': 20}
    es = Elasticsearch(ES_URL)
    # mapping = load_mapping(MAPPING_FILE)
    # recreate_index(es, INDEX_NAME, mapping)
    # bulk_index(es, INDEX_NAME, DATA_FOLDER)
    Fbundle = FeatureBundle(
    essential=Efeatures,
    preferences=Pfeatures,
    quality=Qfeatures,
    functional=Ffeatures,  # or whatever your functional features are
)

    ESCLIENT = ESQueryBuilder(query_mapping, DEFAULT_WEIGHTS)
    QUERY = ESCLIENT.build(Fbundle)
    print("")
    print("=====================================")
    print("Generated ES Query:")
    print(QUERY)
    print("")
    print("=====================================")
    # print(body_2)
    print("")
    print("=====================================")
    example_query(es, INDEX_NAME, QUERY)
    print("=====================================")
