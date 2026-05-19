from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any, Tuple
import os
import json
import glob
import time
import math
import random
import hashlib
import itertools
import traceback
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

print("starting....")

# === FEATURES ===
print("importing feature classes...")
from EA_Features import (
    EssentialFeatures,
    PreferenceFeatures,
    QualityFeatures,
    FeatureBundle,
    UserQuery,
    ModelResult,
)

# === LLM CLIENT ===
print("importing llm client...")
from EB_LLM_Client import LLMClient, LoggingLLMClient

# === FEATURE EXTRACTOR ===
print("importing ALL feature extractors...")
print("importing functional feature extractor...")
from EC_FunctionalFeatureExtractor import NounPhraseExtractor
from EC_FunctionalFeatureExtractor import FunctionalFeatures

print("importing essential feature extractor...")
from EC_EssentialFeatureExtractor import EssentialFeaturesExtractor

print("importing preference feature extractor...")
from EC_PreferenceFeatureExtractor import PreferenceFeaturesExtractor

print("importing quality feature extractor...")
from EC_QualityFeatureExtractor import QualityFeaturesExtractor

print("importing user input...")
from ED_User_Input import user_inputs, rationale_input

print("running ALL feature extractors...")

from E_utils import parse_llm_json_flex, save_to_json, object_to_dict
from EC_PreferenceFeatureExtractor import (
    get_llm_text,
    to_categorical_feat,
    to_numeric_feat,
    to_bool_feat,
    to_recency_feat,
)

from elasticsearch import Elasticsearch


# from EE_Query_Builder_Clean_modified_v2 import ESQueryBuilderAdaptive
# from EE_Query_Builder_Clean_modified_v2 import query_mapping

from EE_Query_Builder_BM25 import ESQueryBuilderAdaptive
from EE_Query_Builder_BM25 import query_mapping

# from EE_Query_Builder_Clean_modified_v3_dedupfix import ESQueryBuilderAdaptive
# from EE_Query_Builder_Clean_modified_v3_dedupfix import query_mapping
# =========================================================
# OPTIONAL VECTOR ENCODER
# =========================================================
_VECTOR_ENCODER = None
_VECTOR_ENCODER_LOAD_FAILED = False


def get_vector_encoder(model_name: str):
    global _VECTOR_ENCODER, _VECTOR_ENCODER_LOAD_FAILED
    if _VECTOR_ENCODER is not None:
        return _VECTOR_ENCODER
    if _VECTOR_ENCODER_LOAD_FAILED:
        return None

    try:
        from sentence_transformers import SentenceTransformer

        print(f"Loading vector encoder: {model_name}")
        _VECTOR_ENCODER = SentenceTransformer(model_name)
        return _VECTOR_ENCODER
    except Exception as e:
        _VECTOR_ENCODER_LOAD_FAILED = True
        print(f"Vector encoder could not be loaded. Falling back to BM25 only. Error: {e}")
        return None


# =========================================================
# HELPERS
# =========================================================


def load_single_feature_json(folder_path, eval_id):
    """
    Loads a single saved eval file like eval_A8.json and returns parsed JSON content.
    """
    pattern = os.path.join(folder_path, f"eval_{eval_id}.json")
    matches = glob.glob(pattern)

    if not matches:
        print(f"File not found: {pattern}")
        return None

    fpath = matches[0]
    with open(fpath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not raw:
        print(f"Empty JSON content in: {fpath}")
        return None

    raw_text = get_llm_text(raw[0])
    return parse_llm_json_flex(raw_text)


# =========================================================
# CONFIG
# =========================================================
#only make recommendaiton
MAKE_RECOMMENDATION = False
ONLY_INTENT = True
EVAL_IDD = "A"
USE_MIXED_EVAL = True
# Mixed-run assignment: map eval prefix -> sample ids
# Example:
EVAL_ID_MAP = {
    # "A": [7, 8, 9],
    "D": [9, 62, 116, 136,177, 212,223],
}
specific_deneme_count = False
OUTPUT_LETTER = "XX_mixed_BM255" # yeniler gden basliyor

ENABLE_RANK_FUNCTIONS = True
ENABLE_QUALITY_DIMENSIONS = True
ENABLE_FEATURE_LOCATIONS = True

INCLUDE_SCORE_BREAKDOWN = False
INCLUDE_EXPLAIN = True
# Rotate this key if it's real
# GEMINI_API_KEY = ""
GEMINI_API_KEY = ""
# GEMINI_API_KEY = ""
# GEMINI_API_KEY = ""

# SPECIFIC_DENEME = {"9"} if specific_deneme_count else None
SPECIFIC_DENEME = {
    "18",  "86", "90", "96",
    "130", "146",
    "178", "179", "184", "213",
    "224",
} if specific_deneme_count else None


# SPECIFIC_DENEME = {"55", "98", "99", "134", "135" } if specific_deneme_count else None # orta iyi olanlr
# SPECIFIC_DENEME = {"68", "69", "89", "157", "159" } # no (backbone popular) icin test
feature_folder_q = "8-CRITERIA_SELECTION/user_intent/quality_features"
feature_folder_e = "8-CRITERIA_SELECTION/user_intent/essential_features"
feature_folder_p = "8-CRITERIA_SELECTION/user_intent/preference_features"

experiment_output_dir = f"8-CRITERIA_SELECTION/experiments/experiment_runs_{OUTPUT_LETTER}"
os.makedirs(experiment_output_dir, exist_ok=True)

ground_truth = {}
non_null_count = 0

ES_URL = "http://localhost:9200"
INDEX_NAME = "models_t7"

# =========================================================
# SEARCH / EXPERIMENT SETTINGS
# =========================================================
# h 486
# c48
# full experiment runliycakstan hyi baska bisey yap eger bir tane deniyceksen extra experimenti true yap
N_EXPERIMENTS =    1
EXTRA_EXPERIMENT = False
EXTRA_EXPERIMENT_NUMBER = 487
RANDOM_SEED = 42
EXPERIMENT_PARALLELISM = max(1, min(8, (os.cpu_count() or 4)))

HYBRID_SEARCH_ENABLED = False
HYBRID_FUSION_MODE = "rrf"
RRF_K = 60
VECTOR_RRF_WEIGHT_DEFAULT = 0.75
BM25_RRF_WEIGHT_DEFAULT = 1.00
VECTOR_ENCODER_MODEL = os.getenv("VECTOR_ENCODER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_FIELD = os.getenv("VECTOR_FIELD", "embedding")
VECTOR_TOP_K = 120
VECTOR_NUM_CANDIDATES = 300

# Balanced baseline
BASE_FEATURE_WEIGHT_GROUPS = {
    "essential": {
        "task": 11.5,
        "domain": 10.5,
        "author": 2.5,
        "objective": 10.0,
        "model_name": 8.0,
    },
    "preference": {
        "license_name": 8.0,
        "library_name": 1.8,
        "basemodels": 10.8,
        "datasets": 1.8,
        "language": 9.5,
        "metrics": 1.0,
    },
    "functional": {
        "functional_item": 12.0,
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
        "likes": 0.8,
        "downloads_last_30_days": 0.75,
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
    "rank": {
        "max": 25.0,
    },
    "match_mode": {
        "grams_factor": 0.90,
    },
}

DEFAULT_PRIORITY_MULTIPLIERS = {
    "must": 1.4,
    "strong_prefer": 1.2,
    "prefer": 1.0,
    "avoid": 0.0,
}

DEFAULT_MINIMUM_SHOULD_MATCH = 5
DEFAULT_SYNONYM_MIN_CONF = 0.20


# # H RUNI 486 tane
# GRID_OPTIONS = {
#     "functional_group_scale": [1, 1.5, 2.0],
#     "essential_group_scale": [1, 1.5, 2.0],
#     "preference_group_scale": [1, 1.5, 2.0],
#     "quality_group_scale": [1, 1.5, 2.0, 4.0, 10.0],  
#     "rank_group_scale": [1, 3, 5],
#     "rank_max": [20, 30, 50, 70],
#     "grams_factor": [0.9],
#     "priority_must": [1.8],
#     "priority_strong_prefer": [1.4],
#     "priority_prefer": [1],
#     "synonym_min_conf": [0.50, 0.70],
#     "minimum_should_match": [1, 3],
#     "target_hits": [150],
#     "size": [100],
#     "vector_rrf_weight": [1.0],
#     "bm25_rrf_weight": [1.0],
# }
# def build_single_experiment_487(number):
#     #exp 111
#     GRID_OPTIONS = {
#         "functional_group_scale": [1],
#         "essential_group_scale": [1],
#         "preference_group_scale": [1],
#         "quality_group_scale": [4.0],
#         "rank_group_scale": [5],
#         "rank_max": [70],
#         "grams_factor": [0.9],
#         "priority_must": [1.8],
#         "priority_strong_prefer": [1.4],
#         "priority_prefer": [1],
#         "synonym_min_conf": [0.5],
#         "minimum_should_match": [1],
#         "target_hits": [50],
#         "size": [30],
#         "vector_rrf_weight": [1.0],
#         "bm25_rrf_weight": [1.0]
#         }

#     return [make_experiment_config(number, GRID_OPTIONS)]

# exp 447 D runi H kirmasi en iyi
GRID_OPTIONS = {
    "functional_group_scale": [2.0],
    "essential_group_scale": [1],
    "preference_group_scale": [1],
    "quality_group_scale": [10.0],
    "rank_group_scale": [5],
    "rank_max": [70],
    "grams_factor": [0.9],
    "priority_must": [1.8],
    "priority_strong_prefer": [1.4],
    "priority_prefer": [1],
    "synonym_min_conf": [0.5],
    "minimum_should_match": [1],
    "target_hits": [50],
    "size": [30],
    "vector_rrf_weight": [1.0],
    "bm25_rrf_weight": [1.0]
}

# EXP 111 -c runi
# GRID_OPTIONS = {
#   "functional_group_scale": [1],
#   "essential_group_scale": [1],
#   "preference_group_scale": [1,1.5, 2],
#   "quality_group_scale": [4.0],
#   "rank_group_scale": [5, 10],
#   "rank_max": [70],
#   "grams_factor": [0.9],
#   "priority_must": [1.4, 1.8],
#   "priority_strong_prefer": [1, 1.4],
#   "priority_prefer": [1],
#   "synonym_min_conf": [0.5 , 0.7],
#   "minimum_should_match": [1],
#   "target_hits": [50],
#   "size": [30],
#   "vector_rrf_weight": [1.0],
#   "bm25_rrf_weight": [1.0]
# }

# # 133
# GRID_OPTIONS = {
#   "functional_group_scale": [1],
#   "essential_group_scale": [1],
#   "preference_group_scale": [1.5],
#   "quality_group_scale": [10.0],
#   "rank_group_scale": [5],
#   "rank_max": [50],
#   "grams_factor": [0.9],
#   "priority_must": [1.8],
#   "priority_strong_prefer": [1.4],
#   "priority_prefer": [1],
#   "synonym_min_conf": [0.7],
#   "minimum_should_match": [1],
#   "target_hits": [50],
#   "size": [30],
#   "vector_rrf_weight": [1.0],
#   "bm25_rrf_weight": [1.0]
# },

FIXED_PARAMS = {
    # "minimum_should_match": 4,
    "target_hits": 50,
    "size": 30

}

TIMING_LOG_TOP_N = 8


# =========================================================
# UTILS
# =========================================================


def round2(x):
    return round(float(x), 4)


def replace_slash(s: str) -> str:
    return s.replace("/", "__")


def scale_dict_values(d, factor):
    return {k: round2(v * factor) for k, v in d.items()}


GROUP_KEYS_FOR_SCALING = ["functional", "essential", "preference", "quality", "rank"]


def deep_copy_config(base_weights, base_boost):
    return deepcopy(base_weights), deepcopy(base_boost)


def apply_group_scales(feature_weights, combo_dict):
    feature_weights["functional"] = scale_dict_values(
        feature_weights["functional"], combo_dict["functional_group_scale"]
    )
    feature_weights["essential"] = scale_dict_values(
        feature_weights["essential"], combo_dict["essential_group_scale"]
    )
    feature_weights["preference"] = scale_dict_values(
        feature_weights["preference"], combo_dict["preference_group_scale"]
    )
    feature_weights["quality"] = scale_dict_values(
        feature_weights["quality"], combo_dict["quality_group_scale"]
    )
    feature_weights["rank"] = scale_dict_values(
        feature_weights["rank"], combo_dict["rank_group_scale"]
    )
    return feature_weights



@dataclass
class ExperimentConfig:
    experiment_id: str
    feature_weight_groups: Dict[str, Dict[str, float]]
    boost_config: Dict[str, Dict[str, float]]
    target_hits: int
    size: int
    minimum_should_match: int = DEFAULT_MINIMUM_SHOULD_MATCH
    synonym_min_conf: float = DEFAULT_SYNONYM_MIN_CONF
    priority_multipliers: Dict[str, float] = field(default_factory=lambda: deepcopy(DEFAULT_PRIORITY_MULTIPLIERS))
    hybrid_search_enabled: bool = HYBRID_SEARCH_ENABLED
    bm25_rrf_weight: float = BM25_RRF_WEIGHT_DEFAULT
    vector_rrf_weight: float = VECTOR_RRF_WEIGHT_DEFAULT
    vector_field: str = VECTOR_FIELD
    vector_top_k: int = VECTOR_TOP_K
    vector_num_candidates: int = VECTOR_NUM_CANDIDATES
    grid_values: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "experiment_id": self.experiment_id,
            "grid_values": self.grid_values,
            "feature_weight_groups": self.feature_weight_groups,
            "boost_config": self.boost_config,
            "target_hits": self.target_hits,
            "size": self.size,
            "minimum_should_match": self.minimum_should_match,
            "synonym_min_conf": self.synonym_min_conf,
            "priority_multipliers": self.priority_multipliers,
            "hybrid_search_enabled": self.hybrid_search_enabled,
            "bm25_rrf_weight": self.bm25_rrf_weight,
            "vector_rrf_weight": self.vector_rrf_weight,
            "vector_field": self.vector_field,
            "vector_top_k": self.vector_top_k,
            "vector_num_candidates": self.vector_num_candidates,
        }



def config_signature(cfg: Dict[str, Any]) -> str:
    payload = deepcopy(cfg)
    payload.pop("experiment_id", None)
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()



def sampled_product(grid_options: Dict[str, List[Any]], limit: int, seed: int = 42):
    """
    Deterministic systematic sampler over the Cartesian product.
    It avoids duplicate experiments and spreads picks across the full grid.
    """
    keys = list(grid_options.keys())
    values = [grid_options[k] for k in keys]
    total = 1
    for v in values:
        total *= len(v)

    if total <= limit:
        for combo in itertools.product(*values):
            combo_dict = dict(zip(keys, combo))
            combo_dict.update(FIXED_PARAMS)
            yield combo_dict
        return

    # Deterministic strided walk across the full Cartesian index space.
    rng = random.Random(seed)
    start = rng.randrange(total)
    step = total // max(1, limit)
    if math.gcd(step, total) != 1:
        step += 1
        while math.gcd(step, total) != 1:
            step += 1

    seen = set()
    idx = start
    produced = 0
    while produced < limit:
        if idx not in seen:
            seen.add(idx)
            combo = []
            rem = idx
            for vals in reversed(values):
                rem, offset = divmod(rem, len(vals))
                combo.append(vals[offset])
            combo.reverse()
            combo_dict = dict(zip(keys, combo))
            combo_dict.update(FIXED_PARAMS)
            yield combo_dict
            produced += 1
        idx = (idx + step) % total



def make_experiment_config(exp_id: int, combo_dict: Dict[str, Any]) -> ExperimentConfig:
    feature_weights, boost_config = deep_copy_config(BASE_FEATURE_WEIGHT_GROUPS, BASE_BOOST_CONFIG)
    feature_weights = apply_group_scales(feature_weights, combo_dict)

    boost_config["rank"]["max"] = round2(combo_dict["rank_max"])
    boost_config["match_mode"]["grams_factor"] = round2(combo_dict["grams_factor"])

    priority_multipliers = {
        "must": round2(combo_dict["priority_must"]),
        "strong_prefer": round2(combo_dict["priority_strong_prefer"]),
        "prefer": round2(combo_dict["priority_prefer"]),
        "avoid": 0.0,
    }

    return ExperimentConfig(
        experiment_id=f"exp_{exp_id:03d}",
        feature_weight_groups=feature_weights,
        boost_config=boost_config,
        target_hits=int(combo_dict["target_hits"]),
        size=int(combo_dict["size"]),
        minimum_should_match=int(combo_dict["minimum_should_match"]),
        synonym_min_conf=round2(combo_dict["synonym_min_conf"]),
        priority_multipliers=priority_multipliers,
        hybrid_search_enabled=HYBRID_SEARCH_ENABLED,
        bm25_rrf_weight=round2(combo_dict["bm25_rrf_weight"]),
        vector_rrf_weight=round2(combo_dict["vector_rrf_weight"]),
        vector_field=VECTOR_FIELD,
        vector_top_k=VECTOR_TOP_K,
        vector_num_candidates=VECTOR_NUM_CANDIDATES,
        grid_values=deepcopy(combo_dict),
    )



def make_grid_experiment_configs(limit: int = 100, seed: int = 42) -> List[ExperimentConfig]:
    configs: List[ExperimentConfig] = []
    signatures = set()

    for idx, combo_dict in enumerate(sampled_product(GRID_OPTIONS, limit=limit, seed=seed), start=1):
        cfg = make_experiment_config(idx, combo_dict)
        sig = config_signature(cfg.to_dict())
        if sig in signatures:
            continue
        signatures.add(sig)
        configs.append(cfg)

    return configs



def build_experiment_configs() -> List[ExperimentConfig]:
    return make_grid_experiment_configs(limit=N_EXPERIMENTS, seed=RANDOM_SEED)



def get_top_hit_model_id(response):
    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        return None
    src = hits[0].get("_source", {}) or {}
    return src.get("modelID") or src.get("model_id") or hits[0].get("_id")



def get_top_k_model_ids(response, k=10):
    hits = response.get("hits", {}).get("hits", [])[:k]
    model_ids = []
    for h in hits:
        src = h.get("_source", {}) or {}
        mid = src.get("modelID") or src.get("model_id") or h.get("_id")
        model_ids.append(mid)
    return model_ids



def get_rank_of_correct_model(response, correct_model, k=100):
    hits = response.get("hits", {}).get("hits", [])[:k]
    for i, h in enumerate(hits, start=1):
        src = h.get("_source", {}) or {}
        mid = src.get("modelID") or src.get("model_id") or h.get("_id")
        if mid == correct_model:
            return i
    return None



def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_stage_timings(stage_timings: Dict[str, float], top_n: int = TIMING_LOG_TOP_N):
    ordered = sorted(stage_timings.items(), key=lambda kv: kv[1], reverse=True)
    parts = [f"{name}={round2(seconds)}s" for name, seconds in ordered[:top_n]]
    return ", ".join(parts)


def build_sample_to_eval_prefix_map(eval_id_map: Dict[str, List[Any]]) -> Dict[str, str]:
    sample_to_prefix: Dict[str, str] = {}
    for prefix, sample_ids in eval_id_map.items():
        normalized_prefix = str(prefix).upper()
        for sample_id in sample_ids:
            sample_key = str(sample_id)
            if sample_key in sample_to_prefix and sample_to_prefix[sample_key] != normalized_prefix:
                raise ValueError(
                    f"Sample {sample_key} is assigned to multiple eval prefixes: "
                    f"{sample_to_prefix[sample_key]} and {normalized_prefix}"
                )
            sample_to_prefix[sample_key] = normalized_prefix
    return sample_to_prefix


def resolve_eval_prefix_for_sample(paper_id: str, default_prefix: str, use_mixed_eval: bool = False) -> str:
    if not use_mixed_eval:
        return str(default_prefix).upper()

    return SAMPLE_TO_EVAL_PREFIX.get(str(paper_id), str(default_prefix).upper())


SAMPLE_TO_EVAL_PREFIX = build_sample_to_eval_prefix_map(EVAL_ID_MAP)


def load_existing_experiment_result(eval_id: str, experiment_id: str):
    """
    Resume helper: if a summary.json already exists for this eval/experiment,
    load and return it so the experiment can be skipped on reruns.
    """
    summary_path = os.path.join(experiment_output_dir, eval_id, experiment_id, "summary.json")
    if not os.path.exists(summary_path):
        return None

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("experiment_id") == experiment_id:
            return data
    except Exception as e:
        print(f"Could not load existing summary for {eval_id}/{experiment_id}: {e}")

    return None


def add_last_modified_aliases(preferences: Any) -> Any:
    recency_value = None
    for attr_name in ("last_modified_year", "last_modified", "lastModified"):
        if hasattr(preferences, attr_name):
            candidate = getattr(preferences, attr_name)
            if candidate is not None:
                recency_value = candidate
                break
    if recency_value is None:
        return preferences

    for attr_name in ("last_modified_year", "last_modified", "lastModified"):
        try:
            setattr(preferences, attr_name, recency_value)
        except Exception:
            pass
    return preferences



def build_query_text(user_text: str, features: FeatureBundle) -> str:
    parts = [user_text]

    try:
        functional = getattr(features, "functional", None)
        func_items = (
            getattr(functional, "F_features", None)
            or getattr(functional, "functional_items", None)
            or getattr(functional, "items", None)
        )
        if func_items:
            parts.extend([str(x) for x in func_items if x])
    except Exception:
        pass

    for group_name in ["essential", "preferences"]:
        group = getattr(features, group_name, None)
        if not group:
            continue
        try:
            raw = object_to_dict(group)
            for _, value in raw.items():
                if value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    parts.append(str(value))
                elif isinstance(value, list):
                    parts.extend([str(v) for v in value if v])
                elif isinstance(value, dict):
                    for _, v in value.items():
                        if v:
                            parts.append(str(v))
        except Exception:
            continue

    return " ".join([p for p in parts if p]).strip()
def normalize_hits(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    return response.get("hits", {}).get("hits", []) if response else []



def make_knn_query(query_vector, vector_field: str, size: int, num_candidates: int, bm25_query: Optional[Dict[str, Any]] = None):
    query = {
        "size": size,
        "knn": {
            "field": vector_field,
            "query_vector": query_vector,
            "k": size,
            "num_candidates": num_candidates,
        },
        "_source": True,
    }

    if bm25_query and isinstance(bm25_query, dict):
        # Use the bm25 query as a filter when supported by ES.
        if "query" in bm25_query and bm25_query["query"]:
            query["knn"]["filter"] = bm25_query["query"]

    return query



def encode_query_text(query_text: str):
    encoder = get_vector_encoder(VECTOR_ENCODER_MODEL)
    if encoder is None:
        return None
    vec = encoder.encode(query_text, normalize_embeddings=True)
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)



def fuse_responses(
    responses: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
    size: int,
    mode: str,
    k: int = 60,
) -> Dict[str, Any]:
    normalized_mode = str(mode or "rrf").strip().lower()
    if normalized_mode != "rrf":
        raise ValueError(f"Unsupported HYBRID_FUSION_MODE={mode!r}. Supported modes: ['rrf']")
    return rrf_fuse_responses(responses=responses, weights=weights, size=size, k=k)


def rrf_fuse_responses(
    responses: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
    size: int,
    k: int = 60,
) -> Dict[str, Any]:
    fused_scores: Dict[str, float] = {}
    hit_by_id: Dict[str, Dict[str, Any]] = {}

    for name, response in responses.items():
        weight = float(weights.get(name, 1.0))
        if weight <= 0:
            continue
        for rank, hit in enumerate(normalize_hits(response), start=1):
            src = hit.get("_source", {}) or {}
            doc_id = src.get("modelID") or src.get("model_id") or hit.get("_id")
            if doc_id is None:
                continue
            fused_scores.setdefault(doc_id, 0.0)
            fused_scores[doc_id] += weight * (1.0 / (k + rank))
            if doc_id not in hit_by_id:
                hit_by_id[doc_id] = hit

    ranked_ids = sorted(fused_scores.keys(), key=lambda d: fused_scores[d], reverse=True)[:size]
    fused_hits = []
    for doc_id in ranked_ids:
        hit = deepcopy(hit_by_id[doc_id])
        hit["_score"] = round2(fused_scores[doc_id])
        fused_hits.append(hit)

    return {
        "hits": {
            "total": {"value": len(fused_scores), "relation": "eq"},
            "hits": fused_hits,
        },
        "_fusion": {
            "mode": "rrf",
            "rrf_k": k,
            "weights": weights,
            "sources": list(responses.keys()),
        },
    }



def instantiate_builder(exp_cfg: ExperimentConfig):
    base_kwargs = dict(
        mapping=query_mapping,
        target_hits=exp_cfg.target_hits,
        size=exp_cfg.size,
        feature_weight_groups=exp_cfg.feature_weight_groups,
        boost_config=exp_cfg.boost_config,
        enable_rank_functions=ENABLE_RANK_FUNCTIONS,
        enable_quality_dimensions=ENABLE_QUALITY_DIMENSIONS,
        enable_feature_locations=ENABLE_FEATURE_LOCATIONS,
        priority_multipliers=exp_cfg.priority_multipliers,
        synonym_min_conf=exp_cfg.synonym_min_conf,
        minimum_should_match=exp_cfg.minimum_should_match,
        synonym_cache_path="synonym_cache.json",
    )

    try:
        return ESQueryBuilderAdaptive(**base_kwargs)
    except TypeError as e:
        raise TypeError(
            "ESQueryBuilderAdaptive constructor mismatch. "
            f"Provided kwargs: {sorted(base_kwargs.keys())}. Original error: {e}"
        ) from e

def run_bm25_and_optional_vector_search(
    exp_cfg: ExperimentConfig,
    features: FeatureBundle,
    user_text: str,
    index_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Any, Dict[str, float], Any, Any]:
    search_stage_timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    es_client = Elasticsearch(ES_URL)
    search_stage_timings["es_client_init_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    builder = instantiate_builder(exp_cfg)
    search_stage_timings["builder_init_search_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    prebuilt_groups = builder.precompute_feature_group_cache(features)
    search_stage_timings["precompute_feature_group_cache_search_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    bm25_response, final_query, final_feature_groups = builder.search(
        es_client=es_client,
        index=index_name,
        features=features,
        prebuilt_groups=prebuilt_groups,
        # include_score_breakdown=INCLUDE_SCORE_BREAKDOWN,
        include_explain=INCLUDE_EXPLAIN,
    )
    search_stage_timings["builder_search_bm25_seconds"] = time.perf_counter() - t0

    hybrid_query = {"bm25": final_query}

    if not exp_cfg.hybrid_search_enabled or exp_cfg.vector_rrf_weight <= 0:
        search_stage_timings["hybrid_total_seconds"] = 0.0
        return bm25_response, hybrid_query, final_feature_groups, search_stage_timings, builder, prebuilt_groups

    hybrid_started = time.perf_counter()

    t0 = time.perf_counter()
    query_text = build_query_text(user_text=user_text, features=features)
    search_stage_timings["build_query_text_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    query_vector = encode_query_text(query_text)
    search_stage_timings["encode_query_text_seconds"] = time.perf_counter() - t0

    if query_vector is None:
        hybrid_query["vector_skipped"] = "encoder_unavailable"
        search_stage_timings["hybrid_total_seconds"] = time.perf_counter() - hybrid_started
        return bm25_response, hybrid_query, final_feature_groups, search_stage_timings, builder, prebuilt_groups

    t0 = time.perf_counter()
    vector_query = make_knn_query(
        query_vector=query_vector,
        vector_field=exp_cfg.vector_field,
        size=min(exp_cfg.size, exp_cfg.vector_top_k),
        num_candidates=exp_cfg.vector_num_candidates,
        bm25_query=final_query,
    )
    search_stage_timings["make_knn_query_seconds"] = time.perf_counter() - t0
    hybrid_query["vector"] = vector_query

    try:
        t0 = time.perf_counter()
        vector_response = es_client.search(index=index_name, body=vector_query)
        search_stage_timings["vector_search_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        fused_response = fuse_responses(
            responses={"bm25": bm25_response, "vector": vector_response},
            weights={"bm25": exp_cfg.bm25_rrf_weight, "vector": exp_cfg.vector_rrf_weight},
            size=exp_cfg.size,
            mode=HYBRID_FUSION_MODE,
            k=RRF_K,
        )
        search_stage_timings["fuse_responses_seconds"] = time.perf_counter() - t0
        search_stage_timings["hybrid_total_seconds"] = time.perf_counter() - hybrid_started
        return fused_response, hybrid_query, final_feature_groups, search_stage_timings, builder, prebuilt_groups
    except Exception as e:
        hybrid_query["vector_error"] = str(e)
        search_stage_timings["hybrid_total_seconds"] = time.perf_counter() - hybrid_started
        return bm25_response, hybrid_query, final_feature_groups, search_stage_timings, builder, prebuilt_groups



def run_single_experiment(
    eval_id: str,
    sample_key: str,
    correct_model: str,
    user_text: str,
    bundle: FeatureBundle,
    exp_cfg: ExperimentConfig,
):
    experiment_id = exp_cfg.experiment_id
    print(f"[{eval_id}] Running {experiment_id}")
    started = time.perf_counter()
    stage_timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    response, final_query, final_feature_groups, search_stage_timings, builder, prebuilt_groups = run_bm25_and_optional_vector_search(
        exp_cfg=exp_cfg,
        features=bundle,
        user_text=user_text,
        index_name=INDEX_NAME,
    )
    stage_timings["search_pipeline_total_seconds"] = time.perf_counter() - t0
    stage_timings.update(search_stage_timings)

    t0 = time.perf_counter()
    top1_model = get_top_hit_model_id(response)
    top10_models = get_top_k_model_ids(response, k=10)
    correct_rank = get_rank_of_correct_model(response, correct_model, k=100)
    stage_timings["result_analysis_seconds"] = time.perf_counter() - t0

    full_output_dir = os.path.join(experiment_output_dir, eval_id, experiment_id)

    t0 = time.perf_counter()
    os.makedirs(full_output_dir, exist_ok=True)
    stage_timings["make_output_dir_seconds"] = time.perf_counter() - t0

    diagnose_error = None
    diagnose = None
    t0 = time.perf_counter()
    try:
        diagnose = builder.compare_bundle_to_sample(
            features=bundle,
            prebuilt_groups=prebuilt_groups,
            sample_file=f"8-CRITERIA_SELECTION/test_models/{replace_slash(correct_model)}.json",
        )
    except Exception as e:
        diagnose_error = {"error": str(e), "traceback": traceback.format_exc()}
    stage_timings["diagnose_seconds"] = time.perf_counter() - t0

    stage_timings["total_elapsed_seconds"] = time.perf_counter() - started

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
        "config": exp_cfg.to_dict(),
        "elapsed_seconds": round2(stage_timings["total_elapsed_seconds"]),
        "stage_timings": {k: round2(v) for k, v in stage_timings.items()},
    }

    t0 = time.perf_counter()
    # save_json_file(os.path.join(full_output_dir, "query.json"), final_query)
    save_json_file(os.path.join(full_output_dir, "response.json"), response)
    save_json_file(os.path.join(full_output_dir, "feature_groups.json"), object_to_dict(final_feature_groups))
    if diagnose_error is None:
        save_json_file(os.path.join(full_output_dir, "diagnose.json"), diagnose)
    else:
        save_json_file(os.path.join(full_output_dir, "diagnose.json"), diagnose_error)
    stage_timings["save_outputs_seconds"] = time.perf_counter() - t0

    stage_timings["total_elapsed_seconds"] = time.perf_counter() - started
    result_row["elapsed_seconds"] = round2(stage_timings["total_elapsed_seconds"])
    result_row["stage_timings"] = {k: round2(v) for k, v in stage_timings.items()}

    t0 = time.perf_counter()
    save_json_file(os.path.join(full_output_dir, "summary.json"), result_row)
    stage_timings["save_summary_seconds"] = time.perf_counter() - t0

    stage_timings["total_elapsed_seconds"] = time.perf_counter() - started
    result_row["elapsed_seconds"] = round2(stage_timings["total_elapsed_seconds"])
    result_row["stage_timings"] = {k: round2(v) for k, v in stage_timings.items()}

    save_json_file(os.path.join(full_output_dir, "summary.json"), result_row)

    print(f"[{eval_id}] {experiment_id} timing breakdown: {format_stage_timings(stage_timings)}")
    return result_row


# =========================================================
# PREPARE EXPERIMENTS
# =========================================================
if not EXTRA_EXPERIMENT:
    experiment_configs = build_experiment_configs()
else:
    experiment_configs = build_single_experiment_487(number=EXTRA_EXPERIMENT_NUMBER)
print(f"Prepared {len(experiment_configs)} unique experiment configurations.")

# Aggregate outputs
all_sample_level_results = []

# =========================================================
# MAIN LOOP OVER PAPERS
# =========================================================
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
    sample_year_1 =original_paper.get("model_year", "unknown_year")
    sample_year = str(int(sample_year_1))
    # sample_year = paper_id.split("-")[0] if "-" in paper_id else "unknown_year"
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

    eval_prefix = resolve_eval_prefix_for_sample(
        paper_id=paper_id,
        default_prefix=EVAL_IDD,
        use_mixed_eval=USE_MIXED_EVAL,
    )
    eval_id = f"{eval_prefix}{paper_id}"
    print(f"\n==============================")
    print(f"Processing {eval_id}")
    print(f"==============================")

    ground_truth[eval_id] = correct_model
    save_json_file("8-CRITERIA_SELECTION/user_intent/ground_truth.json", ground_truth)

    eval_path_q = os.path.join(feature_folder_q, f"eval_{eval_id}.json")
    eval_path_e = os.path.join(feature_folder_e, f"eval_{eval_id}.json")
    eval_path_p = os.path.join(feature_folder_p, f"eval_{eval_id}.json")

    # =====================================================
    # LLM FEATURE EXTRACTION MODE
    # =====================================================
    llm_client = LLMClient(
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.5-flash",
        max_retries=5,
        retry_delay_seconds=20.0,
    )

    Elogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir=feature_folder_e,
        print_output=True,
        save_file=f"eval_{eval_id}.json",
    )
    Plogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir=feature_folder_p,
        print_output=True,
        save_file=f"eval_{eval_id}.json",
    )
    Qlogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir=feature_folder_q,
        print_output=True,
        save_file=f"eval_{eval_id}.json",
    )

    Eextractor = EssentialFeaturesExtractor(Elogger)
    Pextractor = PreferenceFeaturesExtractor(Plogger)
    Qextractor = QualityFeaturesExtractor(Qlogger)

    Ffeatures = FunctionalFeatures()
    Fextractor = NounPhraseExtractor()
    Ffeatures.add_from_query(user_text, Fextractor)

    if not MAKE_RECOMMENDATION:
        if not os.path.exists(eval_path_e):
            print(f"Running EssentialFeaturesExtractor for {sample_key}...")
            time.sleep(10)
            Eextractor.extract(user_text)
        else:
            print(f"Skipping essential extraction, already exists: {eval_path_e}")

        if not os.path.exists(eval_path_p):
            print(f"Running PreferenceFeaturesExtractor for {sample_key}...")
            time.sleep(10)
            Pextractor.extract(user_text)
        else:
            print(f"Skipping preference extraction, already exists: {eval_path_p}")

        if not os.path.exists(eval_path_q):
            print(f"Running QualityFeaturesExtractor for {sample_key}...")
            time.sleep(10)
            Qextractor.extract(user_text)
        else:
            print(f"Skipping quality extraction, already exists: {eval_path_q}")

        if ONLY_INTENT:
            continue

    # =====================================================
    # LOAD SAVED FEATURES ONCE
    # =====================================================
    Efeatures_data = load_single_feature_json(feature_folder_e, eval_id)
    Pfeatures_data = load_single_feature_json(feature_folder_p, eval_id)
    Qfeatures_data = load_single_feature_json(feature_folder_q, eval_id)

    if Efeatures_data is None or Pfeatures_data is None or Qfeatures_data is None:
        print(f"Skipping {eval_id}: one or more feature files missing or invalid.")
        continue

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
        lastModified=to_recency_feat(sample_year),
        # lastModified=to_recency_feat(Pfeatures_data.get("lastModified")),
        library_name=to_categorical_feat(Pfeatures_data.get("library_name")),
        likes=to_numeric_feat(Pfeatures_data.get("likes")),
        tensors_total=to_numeric_feat(Pfeatures_data.get("tensors_total")),
        usedStorage=to_numeric_feat(Pfeatures_data.get("usedStorage")),
        datasets=to_categorical_feat(Pfeatures_data.get("datasets")),
        language=to_categorical_feat(Pfeatures_data.get("language")),
        metrics=to_categorical_feat(Pfeatures_data.get("metrics")),
    )
    Pfeatures = add_last_modified_aliases(Pfeatures)

    print("Creating feature bundle")
    Fbundle = FeatureBundle(
        essential=Efeatures,
        preferences=Pfeatures,
        quality=Qfeatures,
        functional=Ffeatures,
    )

    bundle_output_path = os.path.join(experiment_output_dir, f"{eval_id}_feature_bundle.json")
    save_json_file(bundle_output_path, object_to_dict(Fbundle))

    # =====================================================
    # SEQUENTIAL EXPERIMENTS FOR THIS SAMPLE
    # =====================================================
    sample_experiment_results = []

    for exp_cfg in experiment_configs:
        existing_row = load_existing_experiment_result(eval_id, exp_cfg.experiment_id)
        if existing_row is not None:
            print(f"[{eval_id}] Skipping {exp_cfg.experiment_id} (already completed)")
            sample_experiment_results.append(existing_row)
            all_sample_level_results.append(existing_row)
            continue

        try:
            row = run_single_experiment(
                eval_id,
                sample_key,
                correct_model,
                user_text,
                Fbundle,
                exp_cfg,
            )
            sample_experiment_results.append(row)
            all_sample_level_results.append(row)

        except Exception as e:
            print(f"Experiment failed for {eval_id}: {e}")
            print(traceback.format_exc())

    ranked = sorted(
        sample_experiment_results,
        key=lambda x: (
            999 if x["correct_rank"] is None else x["correct_rank"],
            0 if x["hit_at_1"] else 1,
            x.get("elapsed_seconds", 999999),
        ),
    )

    print(f"\nTop experiment summaries for {eval_id}:")
    for row in ranked[:5]:
        print(
            f"{row['experiment_id']} | "
            f"rank={row['correct_rank']} | "
            f"hit@1={row['hit_at_1']} | "
            f"top1={row['top1_model']} | "
            f"time={row.get('elapsed_seconds')}s"
        )

    save_json_file(
        os.path.join(experiment_output_dir, f"{eval_id}_all_experiments.json"),
        sample_experiment_results,
    )

# =========================================================
# FINAL GLOBAL SUMMARY
# =========================================================
summary_by_experiment = {}

for row in all_sample_level_results:
    exp_id = row["experiment_id"]
    if exp_id not in summary_by_experiment:
        summary_by_experiment[exp_id] = {
            "experiment_id": exp_id,
            "n_samples": 0,
            "hit_at_1_count": 0,
            "hit_at_10_count": 0,
            "rank_sum": 0,
            "rank_count": 0,
            "elapsed_seconds_sum": 0.0,
            "config": row["config"],
        }

    s = summary_by_experiment[exp_id]
    s["n_samples"] += 1
    s["hit_at_1_count"] += int(row["hit_at_1"])
    s["hit_at_10_count"] += int(row["hit_at_10"])
    s["elapsed_seconds_sum"] += float(row.get("elapsed_seconds", 0.0))

    if row["correct_rank"] is not None:
        s["rank_sum"] += row["correct_rank"]
        s["rank_count"] += 1

final_summary = []
for exp_id, s in summary_by_experiment.items():
    n = max(s["n_samples"], 1)
    avg_rank = None if s["rank_count"] == 0 else round(s["rank_sum"] / s["rank_count"], 4)

    final_summary.append({
        "experiment_id": exp_id,
        "n_samples": s["n_samples"],
        "hit_at_1": round(s["hit_at_1_count"] / n, 4),
        "hit_at_10": round(s["hit_at_10_count"] / n, 4),
        "avg_rank_when_found": avg_rank,
        "avg_elapsed_seconds": round(s["elapsed_seconds_sum"] / n, 4),
        "config": s["config"],
    })

final_summary = sorted(
    final_summary,
    key=lambda x: (
        -x["hit_at_1"],
        -x["hit_at_10"],
        999999 if x["avg_rank_when_found"] is None else x["avg_rank_when_found"],
        x.get("avg_elapsed_seconds", 999999),
    ),
)

save_json_file(os.path.join(experiment_output_dir, "global_experiment_summary.json"), final_summary)
save_json_file(os.path.join(experiment_output_dir, "all_sample_level_results.json"), all_sample_level_results)

print(f"\nNUMBER OF TESTED INPUTS: {non_null_count}")
print("\nBest experiments overall:")
for row in final_summary[:10]:
    print(
        f"{row['experiment_id']} | "
        f"hit@1={row['hit_at_1']} | "
        f"hit@10={row['hit_at_10']} | "
        f"avg_rank={row['avg_rank_when_found']} | "
        f"avg_time={row['avg_elapsed_seconds']}"
    )
