import json

# === PREFERENCES AND CONSTRAINTS EXTRACTION MODULES ===
class IntentExtractor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract(self, query: UserQuery) -> IntentFeatures:
        prompt = f"""
        You are helping classify the user's ML needs.
        User query: {query.raw_text}

        Return a JSON object with:
        - task
        - domain
        - objective
        """
        raw = self.llm.call(prompt)
        data = json.loads(raw)
        return IntentFeatures(
            task=data["task"],
            domain=data.get("domain"),
            objective=data.get("objective")
        )


class PreferenceExtractor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract(self, query: UserQuery) -> PreferenceFeatures:
        # similar: enforce JSON schema in the prompt
        ...


class ConstraintExtractor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract(self, query: UserQuery) -> ConstraintFeatures:
        ...

# === ORCHESTRATOR ===


class FeatureOrchestrator:
    def __init__(self,
                 intent_extractor: IntentExtractor,
                 pref_extractor: PreferenceExtractor,
                 constraint_extractor: ConstraintExtractor):
        self.intent_extractor = intent_extractor
        self.pref_extractor = pref_extractor
        self.constraint_extractor = constraint_extractor

    def run(self, query: UserQuery) -> FeatureBundle:
        intent = self.intent_extractor.extract(query)
        prefs = self.pref_extractor.extract(query)
        constraints = self.constraint_extractor.extract(query)
        return FeatureBundle(
            intent=intent,
            preferences=prefs,
            constraints=constraints
        )


# === QUERY BUILDER === 

ES_MAPPING = {
    "task_field": "task",
    "domain_field": "tags",
    "framework_field": "library",
    "language_field": "language",
    "license_field": "license",
    "downloads_field": "downloads",
    "params_field": "model_size_params"
}

QUERY_WEIGHTS = {
    "task_boost": 5.0,
    "domain_boost": 2.0,
    "framework_boost": 1.5,
}

class ESQueryBuilder:
    def __init__(self, mapping: Dict, weights: Dict):
        self.mapping = mapping
        self.weights = weights

    def build(self, features: FeatureBundle) -> Dict:
        must_clauses = []
        filter_clauses = []
        should_clauses = []

        # MUST: task
        must_clauses.append({
            "term": {
                self.mapping["task_field"]: features.intent.task
            }
        })

        # SHOULD: domain
        if features.intent.domain:
            should_clauses.append({
                "term": {
                    self.mapping["domain_field"]: {
                        "value": features.intent.domain,
                        "boost": self.weights["domain_boost"]
                    }
                }
            })

        # SHOULD: frameworks, languages, license
        if features.preferences.frameworks:
            should_clauses.append({
                "terms": {
                    self.mapping["framework_field"]: features.preferences.frameworks
                }
            })

        if features.preferences.languages:
            should_clauses.append({
                "terms": {
                    self.mapping["language_field"]: features.preferences.languages
                }
            })

        if features.preferences.license_preferences:
            should_clauses.append({
                "terms": {
                    self.mapping["license_field"]: features.preferences.license_preferences
                }
            })

        # FILTER: constraints
        if features.constraints.min_downloads:
            filter_clauses.append({
                "range": {
                    self.mapping["downloads_field"]: {
                        "gte": features.constraints.min_downloads
                    }
                }
            })

        if features.constraints.max_model_size_params:
            filter_clauses.append({
                "range": {
                    self.mapping["params_field"]: {
                        "lte": features.constraints.max_model_size_params
                    }
                }
            })

        # Example: license filter for commercial use
        if features.constraints.must_be_commercial_use:
            filter_clauses.append({
                "terms": {
                    self.mapping["license_field"]: ["apache-2.0", "mit", "bsd-3-clause"]
                }
            })

        query = {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses,
                "should": should_clauses,
                "minimum_should_match": 0 if not should_clauses else 1
            }
        }

        return {
            "query": query,
            "size": 20
        }

# ==== ELASTIC SEARCH CLIENT ====

from elasticsearch import Elasticsearch

class ESClient:
    def __init__(self, hosts):
        self.client = Elasticsearch(hosts=hosts)

    def search_models(self, index: str, query_body: Dict) -> List[ModelResult]:
        resp = self.client.search(index=index, body=query_body)
        results = []
        for hit in resp["hits"]["hits"]:
            results.append(ModelResult(
                model_id=hit["_id"],
                score=hit["_score"],
                metadata=hit["_source"],
            ))
        return results


# ==== RECOMMENDER ==== 

class HFModelRecommender:
    def __init__(self,
                 orchestrator: FeatureOrchestrator,
                 query_builder: ESQueryBuilder,
                 es_client: ESClient,
                 index_name: str):
        self.orchestrator = orchestrator
        self.query_builder = query_builder
        self.es_client = es_client
        self.index_name = index_name

    def recommend(self, raw_text: str, user_id: str | None = None) -> List[ModelResult]:
        user_query = UserQuery(raw_text=raw_text, user_id=user_id)
        features = self.orchestrator.run(user_query)
        es_query = self.query_builder.build(features)
        results = self.es_client.search_models(index=self.index_name,
                                               query_body=es_query)
        # optionally rerank here
        return results


# COP: BUNUN AMACI METRICSE FALAN TEK TEK BAKMAK VALUELARININ TYPEI NE DUZGUN VE TEK UNIFORMLAR MI DIYE
import json
import os
from collections import Counter

folder_path = "HF-Models-T6"

def get_ci(d, key):
    key = key.lower()
    for k, v in d.items():
        if isinstance(k, str) and k.lower() == key:
            return v
    return None

top_level_types = Counter()
list_item_types = Counter()

for filename in os.listdir(folder_path):
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data if isinstance(data, list) else [data]

    for item in items:
        if not isinstance(item, dict):
            continue

        md = get_ci(item, "Metadata")
        if not isinstance(md, dict):
            continue

        metrics = get_ci(md, "metrics")
        top_level_types[type(metrics).__name__] += 1

        if isinstance(metrics, list):
            for x in metrics:
                list_item_types[type(x).__name__] += 1

print("Metadata.metrics types:", top_level_types)
print("Types inside metrics lists:", list_item_types)