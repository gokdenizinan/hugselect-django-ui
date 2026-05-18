from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import os
import json
import glob
import time
import random
import itertools
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from math import sqrt

print("starting improved pipeline....")

# === FEATURES ===
print("importing feature classes...")
from EA_Features import (
    EssentialFeatures,
    PreferenceFeatures,
    QualityFeatures,
    FeatureBundle,
)

# === LLM CLIENT ===
print("importing llm client...")
from EB_LLM_Client import LLMClient, LoggingLLMClient

# === FEATURE EXTRACTOR ===
print("importing ALL feature extractors...")
from EC_FunctionalFeatureExtractor import NounPhraseExtractor, FunctionalFeatures
from EC_EssentialFeatureExtractor import EssentialFeaturesExtractor
from EC_PreferenceFeatureExtractor import PreferenceFeaturesExtractor
from EC_QualityFeatureExtractor import QualityFeaturesExtractor
from ED_User_Input import rationale_input
from E_utils import parse_llm_json_flex, object_to_dict
from EC_PreferenceFeatureExtractor import (
    get_llm_text,
    to_categorical_feat,
    to_numeric_feat,
    to_bool_feat,
    to_recency_feat,
)
from elasticsearch import Elasticsearch
from EE_Query_Builder_All_relax_modified import ESQueryBuilderAdaptive, query_mapping


# ---------------------------------------------------------
# Optional semantic reranker backends for hybrid retrieval.
# ---------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    SentenceTransformer = None
    _HAS_ST = False


# =========================================================
# CONFIG
# =========================================================
MAKE_RECOMMENDATION = True
specific_deneme_count = False
OUTPUT_LETTER = "I"

ENABLE_RANK_FUNCTIONS = True
ENABLE_QUALITY_DIMENSIONS = True
ENABLE_FEATURE_LOCATIONS = True
GEMINI_API_KEY = "REMOVED_GEMINI_API_KEY"

SPECIFIC_DENEME = {"8"} if specific_deneme_count else None

feature_folder_q = "8-CRITERIA_SELECTION/user_intent/quality_features"
feature_folder_e = "8-CRITERIA_SELECTION/user_intent/essential_features"
feature_folder_p = "8-CRITERIA_SELECTION/user_intent/preference_features"

experiment_output_dir = f"8-CRITERIA_SELECTION/user_intent/experiment_runs_{OUTPUT_LETTER}"
os.makedirs(experiment_output_dir, exist_ok=True)

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "models_t7")

# Search mode
SEARCH_MODE = "random"
N_EXPERIMENTS = 4
RANDOM_SEED = 42
MAX_WORKERS = min(8, max(1, (os.cpu_count() or 4) - 1))

# Retrieval mode
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid_rrf")
RRF_K = 60
RRF_WEIGHTS = {
    "feature_core": 5.0,
    "feature_expanded": 4.0,
    "preference": 1.5,
    "quality_rank": 0.75,
    "hybrid_semantic": 2.5,
}
HYBRID_CANDIDATE_POOL = 150
HYBRID_RERANK_TOP_K = 100

BASE_FEATURE_WEIGHT_GROUPS = {
    "essential": {
        "task": 8.5,
        "domain": 2.5,
        "author": 2.5,
        "objective": 2.0,
        "model_name": 8.0,
    },
    "preference": {
        "license_name": 1.0,
        "library_name": 1.8,
        "basemodels": 1.8,
        "datasets": 1.8,
        "language": 2.5,
        "metrics": 1.0,
    },
    "functional": {
        "functional_item": 10.0,
    },
    "quality": {
        "Functional_Suitability": 2.0,
        "Compatibility": 1.2,
        "Performance_Efficiency": 1.2,
        "Reliability": 1.2,
        "Interaction_Capability": 1.0,
        "Security": 1.0,
        "Maintainability": 1.0,
        "Flexibility": 1.2,
    },
    "rank": {
        "likes": 0.2,
        "downloads_last_30_days": 0.35,
        "Functional_Suitability": 1.2,
        "Compatibility": 0.8,
        "Performance_Efficiency": 0.8,
        "Reliability": 0.9,
        "Interaction_Capability": 0.7,
        "Security": 0.8,
        "Maintainability": 0.7,
        "Flexibility": 0.8,
    },
}

BASE_BOOST_CONFIG = {
    "tier": {"start": 120.0, "step": 45.0},
    "rank": {"max": 25.0},
    "match_mode": {"grams_factor": 0.90},
}

PARAM_SPACE = {
    "functional_weight": (6.0, 12.0),
    "essential_task_weight": (6.0, 11.0),
    "essential_domain_weight": (1.0, 4.0),
    "essential_objective_weight": (1.0, 4.0),
    "quality_weight_scale": (0.6, 1.4),
    "rank_weight_scale": (0.2, 1.0),
    "tier_start": (80.0, 140.0),
    "tier_step": (25.0, 70.0),
    "rank_max": (15.0, 35.0),
    "grams_factor": (0.85, 0.93),
    "target_hits": (80, 160),
    "size": (50, 120),
}

GRID_OPTIONS = {
    "functional_weight": [4.0, 8.0, 12.0, 16.0],
    "essential_task_weight": [6.0, 10.0, 14.0],
    "quality_weight_scale": [0.75, 1.0, 1.5],
    "tier_start": [80.0, 160.0, 240.0],
    "grams_factor": [0.85, 0.95, 1.0],
}


# =========================================================
# HELPERS
# =========================================================
def round2(x: float) -> float:
    return round(float(x), 4)


def replace_slash(s: str) -> str:
    return s.replace("/", "__")


def save_json_file(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def scale_dict_values(d: Dict[str, float], factor: float) -> Dict[str, float]:
    return {k: round2(v * factor) for k, v in d.items()}


def zero_dict_values(d: Dict[str, float]) -> Dict[str, float]:
    return {k: 0.0 for k in d}


def deep_copy_config(base_weights, base_boost):
    return deepcopy(base_weights), deepcopy(base_boost)


def load_single_feature_json(folder_path: str, eval_id: str):
    pattern = os.path.join(folder_path, f"eval_{eval_id}.json")
    matches = glob.glob(pattern)
    if not matches:
        print(f"File not found: {pattern}")
        return None
    with open(matches[0], "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not raw:
        print(f"Empty JSON content in: {matches[0]}")
        return None
    raw_text = get_llm_text(raw[0])
    return parse_llm_json_flex(raw_text)


def make_random_experiment_config(exp_id, rng):
    feature_weights, boost_config = deep_copy_config(BASE_FEATURE_WEIGHT_GROUPS, BASE_BOOST_CONFIG)
    functional_weight = rng.uniform(*PARAM_SPACE["functional_weight"])
    essential_task_weight = rng.uniform(*PARAM_SPACE["essential_task_weight"])
    essential_domain_weight = rng.uniform(*PARAM_SPACE["essential_domain_weight"])
    essential_objective_weight = rng.uniform(*PARAM_SPACE["essential_objective_weight"])
    quality_scale = rng.uniform(*PARAM_SPACE["quality_weight_scale"])
    rank_scale = rng.uniform(*PARAM_SPACE["rank_weight_scale"])
    tier_start = rng.uniform(*PARAM_SPACE["tier_start"])
    tier_step = rng.uniform(*PARAM_SPACE["tier_step"])
    rank_max = rng.uniform(*PARAM_SPACE["rank_max"])
    grams_factor = rng.uniform(*PARAM_SPACE["grams_factor"])
    target_hits = rng.randint(*PARAM_SPACE["target_hits"])
    size = rng.randint(*PARAM_SPACE["size"])

    feature_weights["functional"]["functional_item"] = round2(functional_weight)
    feature_weights["essential"]["task"] = round2(essential_task_weight)
    feature_weights["essential"]["domain"] = round2(essential_domain_weight)
    feature_weights["essential"]["objective"] = round2(essential_objective_weight)
    feature_weights["quality"] = scale_dict_values(feature_weights["quality"], quality_scale)
    feature_weights["rank"] = scale_dict_values(feature_weights["rank"], rank_scale)

    boost_config["tier"]["start"] = round2(tier_start)
    boost_config["tier"]["step"] = round2(tier_step)
    boost_config["rank"]["max"] = round2(rank_max)
    boost_config["match_mode"]["grams_factor"] = round2(grams_factor)

    return {
        "experiment_id": f"exp_{exp_id:03d}",
        "feature_weight_groups": feature_weights,
        "boost_config": boost_config,
        "target_hits": int(target_hits),
        "size": int(size),
    }


def make_grid_experiment_configs(limit=100):
    keys = list(GRID_OPTIONS.keys())
    values = [GRID_OPTIONS[k] for k in keys]
    configs = []
    for idx, combo in enumerate(itertools.product(*values), start=1):
        if len(configs) >= limit:
            break
        combo_dict = dict(zip(keys, combo))
        feature_weights, boost_config = deep_copy_config(BASE_FEATURE_WEIGHT_GROUPS, BASE_BOOST_CONFIG)
        feature_weights["functional"]["functional_item"] = combo_dict["functional_weight"]
        feature_weights["essential"]["task"] = combo_dict["essential_task_weight"]
        feature_weights["quality"] = scale_dict_values(feature_weights["quality"], combo_dict["quality_weight_scale"])
        boost_config["tier"]["start"] = combo_dict["tier_start"]
        boost_config["match_mode"]["grams_factor"] = combo_dict["grams_factor"]
        configs.append({
            "experiment_id": f"exp_{idx:03d}",
            "feature_weight_groups": feature_weights,
            "boost_config": boost_config,
            "target_hits": 150,
            "size": 100,
        })
    return configs


def build_experiment_configs():
    if SEARCH_MODE == "grid":
        return make_grid_experiment_configs(limit=N_EXPERIMENTS)
    rng = random.Random(RANDOM_SEED)
    return [make_random_experiment_config(i, rng) for i in range(1, N_EXPERIMENTS + 1)]


def get_model_id(hit: Dict[str, Any]) -> Optional[str]:
    src = hit.get("_source", {}) or {}
    return src.get("modelID") or src.get("model_id")


def get_top_hit_model_id(response: Dict[str, Any]) -> Optional[str]:
    hits = response.get("hits", {}).get("hits", [])
    return get_model_id(hits[0]) if hits else None


def get_top_k_model_ids(response: Dict[str, Any], k=10) -> List[str]:
    ids = []
    for hit in response.get("hits", {}).get("hits", [])[:k]:
        model_id = get_model_id(hit)
        if model_id is not None:
            ids.append(model_id)
    return ids


def get_rank_of_correct_model_from_ids(model_ids: List[str], correct_model: str, k=100) -> Optional[int]:
    for i, mid in enumerate(model_ids[:k], start=1):
        if mid == correct_model:
            return i
    return None


def feature_bundle_to_query_text(features: FeatureBundle) -> str:
    chunks: List[str] = []
    functional_items = getattr(features.functional, "functional_items", None) or getattr(features.functional, "items", None) or []
    for item in functional_items:
        if isinstance(item, str):
            chunks.append(item)
        elif isinstance(item, dict):
            chunks.append(str(item.get("value") or item.get("name") or ""))
        else:
            chunks.append(str(item))
    for attr in ["task", "domain", "objective", "model_name", "author"]:
        value = getattr(features.essential, attr, None)
        if value:
            chunks.append(str(value))
    for attr in ["datasets", "language", "metrics", "library_name", "license_name", "basemodels"]:
        value = getattr(features.preferences, attr, None)
        if value:
            chunks.append(str(value))
    return " ".join(x for x in chunks if x).strip()


def hit_to_text(hit: Dict[str, Any]) -> str:
    src = hit.get("_source", {}) or {}
    metadata = src.get("Metadata") or {}
    features = src.get("Features") or {}
    parts = [
        str(src.get("modelID") or src.get("model_id") or ""),
        str(metadata.get("pipeline_tag") or ""),
        " ".join(map(str, metadata.get("tags") or [])),
        json.dumps(features, ensure_ascii=False, default=str),
    ]
    return " ".join(p for p in parts if p).strip()


def tokenize(text: str) -> List[str]:
    return [tok for tok in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if tok]


def simple_similarity(query: str, text: str) -> float:
    q = tokenize(query)
    t = tokenize(text)
    if not q or not t:
        return 0.0
    q_tf = defaultdict(int)
    t_tf = defaultdict(int)
    for tok in q:
        q_tf[tok] += 1
    for tok in t:
        t_tf[tok] += 1
    dot = sum(q_tf[k] * t_tf.get(k, 0) for k in q_tf)
    q_norm = sqrt(sum(v * v for v in q_tf.values()))
    t_norm = sqrt(sum(v * v for v in t_tf.values()))
    if q_norm == 0 or t_norm == 0:
        return 0.0
    return dot / (q_norm * t_norm)


class SemanticReranker:
    def __init__(self):
        self.model = None
        if _HAS_ST:
            try:
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                print("Semantic reranker: sentence-transformers enabled.")
            except Exception as exc:
                print(f"Semantic reranker fallback to token similarity: {exc}")
                self.model = None

    def rank(self, query_text: str, hits: List[Dict[str, Any]]) -> List[str]:
        if not hits:
            return []
        docs = [hit_to_text(hit) for hit in hits]
        ids = [get_model_id(hit) for hit in hits]
        if self.model is None:
            scored = [(mid, simple_similarity(query_text, doc)) for mid, doc in zip(ids, docs) if mid]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [mid for mid, _ in scored]
        query_emb = self.model.encode([query_text], normalize_embeddings=True)[0]
        doc_embs = self.model.encode(docs, normalize_embeddings=True)
        scored = []
        for mid, emb in zip(ids, doc_embs):
            if mid:
                sim = float(sum(a * b for a, b in zip(query_emb, emb)))
                scored.append((mid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in scored]


SEMANTIC_RERANKER = SemanticReranker()


# =========================================================
# RRF + HYBRID RETRIEVAL
# =========================================================
def weighted_rrf(rankings: Dict[str, List[str]], weights: Dict[str, float], k: int = 60) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = defaultdict(float)
    for ranking_name, ranking in rankings.items():
        w = weights.get(ranking_name, 1.0)
        if w == 0 or not ranking:
            continue
        seen = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += w * (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def response_from_ranked_ids(ranked_ids: List[str], source_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    hits = []
    for pos, model_id in enumerate(ranked_ids, start=1):
        source_hit = deepcopy(source_by_id.get(model_id, {}))
        if not source_hit:
            source_hit = {"_source": {"modelID": model_id}, "_score": 0.0}
        source_hit["_score"] = float(len(ranked_ids) - pos + 1)
        hits.append(source_hit)
    return {"hits": {"total": {"value": len(hits)}, "hits": hits}}


def make_strategy_feature_weights(base_weights: Dict[str, Dict[str, float]], strategy_name: str) -> Dict[str, Dict[str, float]]:
    out = deepcopy(base_weights)
    if strategy_name == "feature_core":
        out["preference"] = zero_dict_values(out["preference"])
        out["quality"] = zero_dict_values(out["quality"])
        out["rank"] = zero_dict_values(out["rank"])
    elif strategy_name == "feature_expanded":
        out["quality"] = zero_dict_values(out["quality"])
        out["rank"] = zero_dict_values(out["rank"])
    elif strategy_name == "preference":
        out["functional"] = zero_dict_values(out["functional"])
        out["essential"] = zero_dict_values(out["essential"])
        out["quality"] = zero_dict_values(out["quality"])
        out["rank"] = zero_dict_values(out["rank"])
    elif strategy_name == "quality_rank":
        out["functional"] = zero_dict_values(out["functional"])
        out["essential"] = zero_dict_values(out["essential"])
        out["preference"] = zero_dict_values(out["preference"])
    return out


def run_single_strategy(es_client: Elasticsearch, exp_cfg: Dict[str, Any], features: FeatureBundle, strategy_name: str) -> Dict[str, Any]:
    strategy_weights = make_strategy_feature_weights(exp_cfg["feature_weight_groups"], strategy_name)
    builder = ESQueryBuilderAdaptive(
        mapping=query_mapping,
        target_hits=max(exp_cfg["target_hits"], HYBRID_CANDIDATE_POOL),
        size=max(exp_cfg["size"], HYBRID_CANDIDATE_POOL),
        feature_weight_groups=strategy_weights,
        boost_config=exp_cfg["boost_config"],
        enable_rank_functions=ENABLE_RANK_FUNCTIONS,
        enable_quality_dimensions=ENABLE_QUALITY_DIMENSIONS,
        enable_feature_locations=ENABLE_FEATURE_LOCATIONS,
    )
    response, final_query, final_feature_groups = builder.search(
        es_client=es_client,
        index=INDEX_NAME,
        features=features,
        include_score_breakdown=True,
        include_explain=False,
    )
    return {
        "strategy_name": strategy_name,
        "response": response,
        "query": final_query,
        "feature_groups": final_feature_groups,
        "builder": builder,
    }


def run_hybrid_rrf(es_client: Elasticsearch, exp_cfg: Dict[str, Any], features: FeatureBundle) -> Dict[str, Any]:
    strategy_names = ["feature_core", "feature_expanded", "preference", "quality_rank"]
    strategy_outputs = [run_single_strategy(es_client, exp_cfg, features, name) for name in strategy_names]

    rankings: Dict[str, List[str]] = {}
    source_by_id: Dict[str, Dict[str, Any]] = {}
    for output in strategy_outputs:
        model_ids = get_top_k_model_ids(output["response"], k=HYBRID_CANDIDATE_POOL)
        rankings[output["strategy_name"]] = model_ids
        for hit in output["response"].get("hits", {}).get("hits", []):
            mid = get_model_id(hit)
            if mid and mid not in source_by_id:
                source_by_id[mid] = hit

    candidate_ids = [doc_id for doc_id, _ in weighted_rrf(rankings, RRF_WEIGHTS, k=RRF_K)][:HYBRID_CANDIDATE_POOL]
    candidate_hits = [source_by_id[mid] for mid in candidate_ids if mid in source_by_id]

    query_text = feature_bundle_to_query_text(features)
    semantic_ranking = SEMANTIC_RERANKER.rank(query_text, candidate_hits)[:HYBRID_RERANK_TOP_K]
    rankings["hybrid_semantic"] = semantic_ranking

    final_ranked_ids = [doc_id for doc_id, _ in weighted_rrf(rankings, RRF_WEIGHTS, k=RRF_K)]
    fused_response = response_from_ranked_ids(final_ranked_ids, source_by_id)

    # Keep the feature-core builder for diagnostics.
    primary_output = next(o for o in strategy_outputs if o["strategy_name"] == "feature_core")
    return {
        "response": fused_response,
        "query": {
            "retrieval_mode": "hybrid_rrf",
            "strategies": {o["strategy_name"]: o["query"] for o in strategy_outputs},
            "rrf_k": RRF_K,
            "rrf_weights": RRF_WEIGHTS,
            "hybrid_query_text": query_text,
        },
        "feature_groups": {o["strategy_name"]: object_to_dict(o["feature_groups"]) for o in strategy_outputs},
        "diagnostic_builder": primary_output["builder"],
        "component_rankings": rankings,
    }


def run_search(es_client: Elasticsearch, exp_cfg: Dict[str, Any], features: FeatureBundle) -> Dict[str, Any]:
    if RETRIEVAL_MODE == "single_es":
        output = run_single_strategy(es_client, exp_cfg, features, "feature_expanded")
        return {
            "response": output["response"],
            "query": output["query"],
            "feature_groups": object_to_dict(output["feature_groups"]),
            "diagnostic_builder": output["builder"],
            "component_rankings": {"feature_expanded": get_top_k_model_ids(output["response"], k=HYBRID_CANDIDATE_POOL)},
        }
    return run_hybrid_rrf(es_client, exp_cfg, features)


# =========================================================
# SAMPLE PREPARATION
# =========================================================
def build_feature_bundle(eval_id: str, user_text: str) -> Optional[FeatureBundle]:
    eval_path_q = os.path.join(feature_folder_q, f"eval_{eval_id}.json")
    eval_path_e = os.path.join(feature_folder_e, f"eval_{eval_id}.json")
    eval_path_p = os.path.join(feature_folder_p, f"eval_{eval_id}.json")

    llm_client = LLMClient(
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.5-flash",
        max_retries=5,
        retry_delay_seconds=20.0,
    )
    Elogger = LoggingLLMClient(llm_client=llm_client, save_dir=feature_folder_e, print_output=True, save_file=f"eval_{eval_id}.json")
    Plogger = LoggingLLMClient(llm_client=llm_client, save_dir=feature_folder_p, print_output=True, save_file=f"eval_{eval_id}.json")
    Qlogger = LoggingLLMClient(llm_client=llm_client, save_dir=feature_folder_q, print_output=True, save_file=f"eval_{eval_id}.json")

    Eextractor = EssentialFeaturesExtractor(Elogger)
    Pextractor = PreferenceFeaturesExtractor(Plogger)
    Qextractor = QualityFeaturesExtractor(Qlogger)

    Ffeatures = FunctionalFeatures()
    Fextractor = NounPhraseExtractor()
    Ffeatures.add_from_query(user_text, Fextractor)

    if not MAKE_RECOMMENDATION:
        if not os.path.exists(eval_path_e):
            time.sleep(10)
            Eextractor.extract(user_text)
        if not os.path.exists(eval_path_p):
            time.sleep(10)
            Pextractor.extract(user_text)
        if not os.path.exists(eval_path_q):
            time.sleep(10)
            Qextractor.extract(user_text)
        return None

    Efeatures_data = load_single_feature_json(feature_folder_e, eval_id)
    Pfeatures_data = load_single_feature_json(feature_folder_p, eval_id)
    Qfeatures_data = load_single_feature_json(feature_folder_q, eval_id)
    if Efeatures_data is None or Pfeatures_data is None or Qfeatures_data is None:
        print(f"Skipping {eval_id}: one or more feature files missing or invalid.")
        return None

    Efeatures = EssentialFeatures(
        task=to_categorical_feat(Efeatures_data.get("task")),
        domain=to_categorical_feat(Efeatures_data.get("domain")),
        model_name=to_categorical_feat(Efeatures_data.get("model_name")),
        author=to_categorical_feat(Efeatures_data.get("author")),
        objective=to_categorical_feat(Efeatures_data.get("objective")),
        task_aliases=to_categorical_feat(Efeatures_data.get("task_aliases")),
        domain_aliases=to_categorical_feat(Efeatures_data.get("domain_aliases")),
    )
    Qfeatures = QualityFeatures(
        Functional_Suitability=Qfeatures_data.get("Functional_Suitability"),
        Compatibility=Qfeatures_data.get("Compatibility"),
        Performance_Efficiency=Qfeatures_data.get("Performance_Efficiency"),
        Reliability=Qfeatures_data.get("Reliability"),
        Interaction_Capability=Qfeatures_data.get("Interaction_Capability"),
        Security=Qfeatures_data.get("Security"),
        Maintainability=Qfeatures_data.get("Maintainability"),
        Flexibility=Qfeatures_data.get("Flexibility"),
    )
    Pfeatures = PreferenceFeatures(
        basemodels=to_categorical_feat(Pfeatures_data.get("basemodels")),
        license_name=to_categorical_feat(Pfeatures_data.get("license_name")),
        downloads_all_time=to_numeric_feat(Pfeatures_data.get("downloads_all_time")),
        downloads_last_30_days=to_numeric_feat(Pfeatures_data.get("downloads_last_30_days")),
        file_count=to_numeric_feat(Pfeatures_data.get("file_count")),
        gated=to_bool_feat(Pfeatures_data.get("gated")),
        lastModified=to_recency_feat(Pfeatures_data.get("lastModified")),
        library_name=to_categorical_feat(Pfeatures_data.get("library_name")),
        likes=to_numeric_feat(Pfeatures_data.get("likes")),
        tensors_total=to_numeric_feat(Pfeatures_data.get("tensors_total")),
        usedStorage=to_numeric_feat(Pfeatures_data.get("usedStorage")),
        datasets=to_categorical_feat(Pfeatures_data.get("datasets")),
        language=to_categorical_feat(Pfeatures_data.get("language")),
        metrics=to_categorical_feat(Pfeatures_data.get("metrics")),
    )
    return FeatureBundle(essential=Efeatures, preferences=Pfeatures, quality=Qfeatures, functional=Ffeatures)


# =========================================================
# PARALLEL EXPERIMENT EXECUTION
# =========================================================
def run_one_experiment(es_url: str, exp_cfg: Dict[str, Any], features: FeatureBundle, eval_id: str, sample_key: str, correct_model: str) -> Dict[str, Any]:
    experiment_id = exp_cfg["experiment_id"]
    print(f"[{eval_id}] Running {experiment_id} with {RETRIEVAL_MODE}")
    es_client = Elasticsearch(es_url)
    search_output = run_search(es_client, exp_cfg, features)
    response = search_output["response"]
    ranked_ids = get_top_k_model_ids(response, k=100)
    top1_model = ranked_ids[0] if ranked_ids else None
    top10_models = ranked_ids[:10]
    correct_rank = get_rank_of_correct_model_from_ids(ranked_ids, correct_model, k=100)

    result_row = {
        "eval_id": eval_id,
        "sample_key": sample_key,
        "experiment_id": experiment_id,
        "correct_model": correct_model,
        "top1_model": top1_model,
        "top10_models": top10_models,
        "correct_rank": correct_rank,
        "hit_at_1": top1_model == correct_model,
        "hit_at_10": correct_model in top10_models,
        "config": exp_cfg,
    }

    full_output_dir = os.path.join(experiment_output_dir, eval_id, experiment_id)
    save_json_file(os.path.join(full_output_dir, "query.json"), search_output["query"])
    save_json_file(os.path.join(full_output_dir, "response.json"), response)
    save_json_file(os.path.join(full_output_dir, "feature_groups.json"), search_output["feature_groups"])
    save_json_file(os.path.join(full_output_dir, "summary.json"), result_row)
    save_json_file(os.path.join(full_output_dir, "component_rankings.json"), search_output["component_rankings"])

    builder = search_output["diagnostic_builder"]
    if builder is not None and hasattr(builder, "compare_fundle_to_sample"):
        try:
            diagnose = builder.compare_fundle_to_sample(
                features=features,
                sample_file=f"8-CRITERIA_SELECTION/test_models/{replace_slash(correct_model)}.json",
            )
            save_json_file(os.path.join(full_output_dir, "diagnose.json"), diagnose)
        except Exception as exc:
            save_json_file(os.path.join(full_output_dir, "diagnose.json"), {"error": str(exc)})

    return result_row


# =========================================================
# MAIN
# =========================================================
def main():
    experiment_configs = build_experiment_configs()
    print(f"Prepared {len(experiment_configs)} experiment configurations.")

    all_sample_level_results: List[Dict[str, Any]] = []
    ground_truth: Dict[str, str] = {}
    non_null_count = 0

    for original_key, original_paper in rationale_input.items():
        if not original_paper:
            continue
        parts = original_key.split("_")
        if len(parts) < 2:
            print(f"Skipping malformed key: {original_key}")
            continue
        paper_id = parts[1]
        if SPECIFIC_DENEME is not None and paper_id not in SPECIFIC_DENEME:
            continue

        sample_key = f"sample_{paper_id}"
        paper = rationale_input.get(sample_key)
        if not paper:
            print(f"Skipping {sample_key}: no data found.")
            continue

        user_text = paper.get("user_intent")
        correct_model = paper.get("model_full_name")
        if user_text is None or correct_model is None:
            print(f"Skipping {sample_key}: missing user_intent or model_full_name.")
            continue

        non_null_count += 1
        eval_id = f"A{paper_id}"
        ground_truth[eval_id] = correct_model
        save_json_file("8-CRITERIA_SELECTION/user_intent/ground_truth.json", ground_truth)

        print("\n==============================")
        print(f"Processing {eval_id}")
        print("==============================")

        bundle = build_feature_bundle(eval_id, user_text)
        if bundle is None:
            continue

        bundle_output_path = os.path.join(experiment_output_dir, f"{eval_id}_feature_bundle.json")
        save_json_file(bundle_output_path, object_to_dict(bundle))

        sample_experiment_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(experiment_configs))) as executor:
            futures = {
                executor.submit(run_one_experiment, ES_URL, exp_cfg, bundle, eval_id, sample_key, correct_model): exp_cfg
                for exp_cfg in experiment_configs
            }
            for future in as_completed(futures):
                result_row = future.result()
                sample_experiment_results.append(result_row)
                all_sample_level_results.append(result_row)

        ranked = sorted(
            sample_experiment_results,
            key=lambda x: (999 if x["correct_rank"] is None else x["correct_rank"], 0 if x["hit_at_1"] else 1),
        )
        print(f"\nTop experiment summaries for {eval_id}:")
        for row in ranked[:5]:
            print(f"{row['experiment_id']} | rank={row['correct_rank']} | hit@1={row['hit_at_1']} | top1={row['top1_model']}")
        save_json_file(os.path.join(experiment_output_dir, f"{eval_id}_all_experiments.json"), sample_experiment_results)

    summary_by_experiment: Dict[str, Dict[str, Any]] = {}
    for row in all_sample_level_results:
        exp_id = row["experiment_id"]
        summary_by_experiment.setdefault(exp_id, {
            "experiment_id": exp_id,
            "n_samples": 0,
            "hit_at_1_count": 0,
            "hit_at_10_count": 0,
            "rank_sum": 0,
            "rank_count": 0,
            "config": row["config"],
        })
        s = summary_by_experiment[exp_id]
        s["n_samples"] += 1
        s["hit_at_1_count"] += int(row["hit_at_1"])
        s["hit_at_10_count"] += int(row["hit_at_10"])
        if row["correct_rank"] is not None:
            s["rank_sum"] += row["correct_rank"]
            s["rank_count"] += 1

    final_summary: List[Dict[str, Any]] = []
    for exp_id, s in summary_by_experiment.items():
        n = max(s["n_samples"], 1)
        avg_rank = None if s["rank_count"] == 0 else round(s["rank_sum"] / s["rank_count"], 4)
        final_summary.append({
            "experiment_id": exp_id,
            "n_samples": s["n_samples"],
            "hit_at_1": round(s["hit_at_1_count"] / n, 4),
            "hit_at_10": round(s["hit_at_10_count"] / n, 4),
            "avg_rank_when_found": avg_rank,
            "config": s["config"],
        })

    final_summary.sort(key=lambda x: (-x["hit_at_1"], -x["hit_at_10"], 999999 if x["avg_rank_when_found"] is None else x["avg_rank_when_found"]))
    save_json_file(os.path.join(experiment_output_dir, "global_experiment_summary.json"), final_summary)
    save_json_file(os.path.join(experiment_output_dir, "all_sample_level_results.json"), all_sample_level_results)

    print(f"\nNUMBER OF TESTED INPUTS: {non_null_count}")
    print("\nBest experiments overall:")
    for row in final_summary[:10]:
        print(f"{row['experiment_id']} | hit@1={row['hit_at_1']} | hit@10={row['hit_at_10']} | avg_rank={row['avg_rank_when_found']}")


if __name__ == "__main__":
    main()
