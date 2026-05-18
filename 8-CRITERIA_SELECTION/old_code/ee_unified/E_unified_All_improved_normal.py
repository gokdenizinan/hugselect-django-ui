from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any, Tuple
import os
import glob
import json
import time
import random
import itertools
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
from EE_Query_Builder_All_relax_modified import ESQueryBuilderAdaptive
from EE_Query_Builder_All_relax_modified import query_mapping


# =========================================================
# HELPERS: FEATURE LOADING
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
MAKE_RECOMMENDATION = True
specific_deneme_count = False
OUTPUT_LETTER = "II"

ENABLE_RANK_FUNCTIONS = True
ENABLE_QUALITY_DIMENSIONS = True
ENABLE_FEATURE_LOCATIONS = True

# Prefer environment variable over hard-coded secret.
GEMINI_API_KEY = "REMOVED_GEMINI_API_KEY"
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set. LLM extraction mode will fail unless cached feature files already exist.")

SPECIFIC_DENEME = {"8"} if specific_deneme_count else None

feature_folder_q = "8-CRITERIA_SELECTION/user_intent/quality_features"
feature_folder_e = "8-CRITERIA_SELECTION/user_intent/essential_features"
feature_folder_p = "8-CRITERIA_SELECTION/user_intent/preference_features"

experiment_output_dir = f"8-CRITERIA_SELECTION/user_intent/experiment_runs_{OUTPUT_LETTER}"
os.makedirs(experiment_output_dir, exist_ok=True)

ground_truth = {}
non_null_count = 0

ES_URL = "http://localhost:9200"
INDEX_NAME = "models_t7"

# =========================================================
# EXPERIMENT SETTINGS
# =========================================================
SEARCH_MODE = "random"   # or "grid"
N_EXPERIMENTS = 4
RANDOM_SEED = 42

# New: parallel experiment execution.
MAX_EXPERIMENT_WORKERS = max(1, min(8, (os.cpu_count() or 4)))
MAX_SIGNAL_WORKERS = 4  # parallel searches *inside* one experiment when RRF is enabled

# New: weighted RRF configuration.
USE_WEIGHTED_RRF = True
RRF_K = 60
# Feature-heavy weights. Functional + essential dominate.
RRF_WEIGHTS = {
    "functional": 5.0,
    "essential": 4.0,
    "preference": 1.5,
    "quality_rank": 0.6,
}
# Optional candidate gating from the feature-centric lists.
RRF_LIMIT_PER_SIGNAL = 150
RRF_FEATURE_GATE_TOPN = 150

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
    "tier": {
        "start": 120.0,
        "step": 45.0,
    },
    "rank": {
        "max": 25.0,
    },
    "match_mode": {
        "grams_factor": 0.90,
    },
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
    "minimum_should_match": (1, 5),
    "size": (50, 120),
}

GRID_OPTIONS = {
    "functional_weight": [4.0, 8.0, 12.0, 16.0],
    "essential_task_weight": [6.0, 10.0, 14.0],
    "quality_weight_scale": [0.75, 1.0, 1.5],
    "tier_start": [80.0, 160.0, 240.0],
    "grams_factor": [0.85, 0.95, 1.0],
    "minimum_should_match": [1, 2],
}


# =========================================================
# GENERIC HELPERS
# =========================================================
def round2(x):
    return round(float(x), 4)


def replace_slash(s: str) -> str:
    return s.replace("/", "__")


def scale_dict_values(d, factor):
    out = {}
    for k, v in d.items():
        out[k] = round2(v * factor)
    return out


def deep_copy_config(base_weights, base_boost):
    return deepcopy(base_weights), deepcopy(base_boost)


def zero_out_weight_groups(weights: Dict[str, Dict[str, float]], keep_groups: List[str]) -> Dict[str, Dict[str, float]]:
    keep = set(keep_groups)
    out = deepcopy(weights)
    for group_name, group_values in out.items():
        if group_name in keep:
            continue
        for feature_name in list(group_values.keys()):
            group_values[feature_name] = 0.0
    return out


def make_random_experiment_config(exp_id, rng):
    feature_weights, boost_config = deep_copy_config(
        BASE_FEATURE_WEIGHT_GROUPS,
        BASE_BOOST_CONFIG,
    )

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
        feature_weights, boost_config = deep_copy_config(
            BASE_FEATURE_WEIGHT_GROUPS,
            BASE_BOOST_CONFIG,
        )

        feature_weights["functional"]["functional_item"] = combo_dict["functional_weight"]
        feature_weights["essential"]["task"] = combo_dict["essential_task_weight"]
        feature_weights["quality"] = scale_dict_values(
            feature_weights["quality"],
            combo_dict["quality_weight_scale"],
        )
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


def new_es_client() -> Elasticsearch:
    # Separate client per worker is the safest option when parallelizing.
    return Elasticsearch(ES_URL)


def save_json_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_model_id_from_hit(hit) -> Optional[str]:
    src = hit.get("_source", {}) or {}
    return src.get("modelID") or src.get("model_id")


def get_top_hit_model_id(response):
    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        return None
    return get_model_id_from_hit(hits[0])


def get_top_k_model_ids(response, k=10):
    hits = response.get("hits", {}).get("hits", [])[:k]
    return [get_model_id_from_hit(h) for h in hits if get_model_id_from_hit(h)]


def get_rank_of_correct_model(response, correct_model, k=100):
    hits = response.get("hits", {}).get("hits", [])[:k]
    for i, h in enumerate(hits, start=1):
        mid = get_model_id_from_hit(h)
        if mid == correct_model:
            return i
    return None


def get_rank_of_correct_model_from_ids(model_ids: List[str], correct_model: str, k: int = 100):
    for i, mid in enumerate(model_ids[:k], start=1):
        if mid == correct_model:
            return i
    return None


def print_results(res, limit=5):
    print("\nModels Matching Query:")
    hits_obj = res.get("hits", {}).get("hits", [])[:limit]

    for hit in hits_obj:
        print("====  ====  ====  ====  ====. ====  ====  ====  ====  ====")

        score = hit.get("_score", hit.get("score"))
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "None"

        src = hit.get("_source", {}) or {}
        model_id = src.get("modelID") or src.get("model_id")
        tags = ((src.get("Metadata") or {}).get("tags"))
        features = src.get("Features")
        pipeline_tag = ((src.get("Metadata") or {}).get("pipeline_tag"))

        print(
            f"- score={score_str}, \n"
            f"modelID={model_id!r}, \n"
            f"=    =    =    =    =    =     = \n"
            f"pipeline_tag={pipeline_tag!r}, \n"
            f"tags={tags!r}, \n"
            f"Features={features!r} \n"
        )


# =========================================================
# WEIGHTED RRF HELPERS
# =========================================================
def build_signal_experiment_configs(exp_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build smaller, specialized configs so different signals can each retrieve
    their own ranking list. This is then fused with weighted RRF.
    """
    base = deepcopy(exp_cfg)
    base_weights = base["feature_weight_groups"]

    signal_cfgs = {
        "functional": deepcopy(base),
        "essential": deepcopy(base),
        "preference": deepcopy(base),
        "quality_rank": deepcopy(base),
    }

    signal_cfgs["functional"]["feature_weight_groups"] = zero_out_weight_groups(base_weights, ["functional"])
    signal_cfgs["essential"]["feature_weight_groups"] = zero_out_weight_groups(base_weights, ["essential"])
    signal_cfgs["preference"]["feature_weight_groups"] = zero_out_weight_groups(base_weights, ["preference"])
    signal_cfgs["quality_rank"]["feature_weight_groups"] = zero_out_weight_groups(base_weights, ["quality", "rank"])

    for name, cfg in signal_cfgs.items():
        cfg["size"] = max(cfg.get("size", 100), RRF_LIMIT_PER_SIGNAL)
        cfg["signal_name"] = name

    return signal_cfgs


def run_search_for_config(
    exp_cfg: Dict[str, Any],
    features: FeatureBundle,
) -> Dict[str, Any]:
    """
    One ES search using one configuration. Safe to call in worker threads.
    """
    es = new_es_client()
    builder = ESQueryBuilderAdaptive(
        mapping=query_mapping,
        target_hits=exp_cfg["target_hits"],
        size=exp_cfg["size"],
        feature_weight_groups=exp_cfg["feature_weight_groups"],
        boost_config=exp_cfg["boost_config"],
        enable_rank_functions=ENABLE_RANK_FUNCTIONS,
        enable_quality_dimensions=ENABLE_QUALITY_DIMENSIONS,
        enable_feature_locations=ENABLE_FEATURE_LOCATIONS,
    )

    response, final_query, final_feature_groups = builder.search(
        es_client=es,
        index=INDEX_NAME,
        features=features,
        include_score_breakdown=True,
        include_explain=False,
    )

    ranking_ids = get_top_k_model_ids(response, k=RRF_LIMIT_PER_SIGNAL)
    return {
        "signal_name": exp_cfg.get("signal_name", "unified"),
        "response": response,
        "query": final_query,
        "feature_groups": object_to_dict(final_feature_groups),
        "ranking_ids": ranking_ids,
    }


def weighted_rrf_fuse(
    rankings: Dict[str, List[str]],
    weights: Dict[str, float],
    k: int = 60,
    gate_candidates_from: Optional[List[str]] = None,
    gate_topn: Optional[int] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Weighted RRF: high-priority rankings contribute more.
    Optionally gates the candidate set using feature-centric signals only.
    """
    allowed_candidates = None
    if gate_candidates_from:
        allowed_candidates = set()
        for ranking_name in gate_candidates_from:
            ranked_ids = rankings.get(ranking_name, [])
            if gate_topn is not None:
                ranked_ids = ranked_ids[:gate_topn]
            allowed_candidates.update(ranked_ids)

    scores: Dict[str, float] = {}
    contributions: Dict[str, Dict[str, float]] = {}

    for ranking_name, model_ids in rankings.items():
        weight = float(weights.get(ranking_name, 0.0))
        if weight <= 0:
            continue

        for rank, model_id in enumerate(model_ids, start=1):
            if not model_id:
                continue
            if allowed_candidates is not None and model_id not in allowed_candidates:
                continue

            contrib = weight * (1.0 / (k + rank))
            scores[model_id] = scores.get(model_id, 0.0) + contrib
            contributions.setdefault(model_id, {})[ranking_name] = contrib

    fused = [mid for mid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return fused, {
        "k": k,
        "weights": weights,
        "allowed_candidates_count": None if allowed_candidates is None else len(allowed_candidates),
        "scores": scores,
        "contributions": contributions,
    }


def run_single_experiment(
    eval_id: str,
    sample_key: str,
    correct_model: str,
    features: FeatureBundle,
    exp_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run one experiment. Uses weighted RRF when enabled; otherwise uses the
    original single-query approach.
    """
    experiment_id = exp_cfg["experiment_id"]
    print(f"[{eval_id}] Running {experiment_id}")

    if not USE_WEIGHTED_RRF:
        single = run_search_for_config(exp_cfg, features)
        response = single["response"]
        top1_model = get_top_hit_model_id(response)
        top10_models = get_top_k_model_ids(response, k=10)
        correct_rank = get_rank_of_correct_model(response, correct_model, k=100)

        return {
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
            "mode": "unified",
            "artifacts": {
                "query": single["query"],
                "response": response,
                "feature_groups": single["feature_groups"],
            },
        }

    # --- Weighted RRF path ---
    signal_cfgs = build_signal_experiment_configs(exp_cfg)
    signal_results: Dict[str, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=min(MAX_SIGNAL_WORKERS, len(signal_cfgs))) as executor:
        future_to_name = {
            executor.submit(run_search_for_config, cfg, features): name
            for name, cfg in signal_cfgs.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            signal_results[name] = future.result()

    rankings = {name: data["ranking_ids"] for name, data in signal_results.items()}
    fused_ids, rrf_debug = weighted_rrf_fuse(
        rankings=rankings,
        weights=RRF_WEIGHTS,
        k=RRF_K,
        gate_candidates_from=["functional", "essential"],
        gate_topn=RRF_FEATURE_GATE_TOPN,
    )

    top1_model = fused_ids[0] if fused_ids else None
    top10_models = fused_ids[:10]
    correct_rank = get_rank_of_correct_model_from_ids(fused_ids, correct_model, k=100)

    return {
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
        "mode": "weighted_rrf",
        "rrf_weights": deepcopy(RRF_WEIGHTS),
        "rrf_k": RRF_K,
        "artifacts": {
            "signal_queries": {name: data["query"] for name, data in signal_results.items()},
            "signal_responses": {name: data["response"] for name, data in signal_results.items()},
            "signal_feature_groups": {name: data["feature_groups"] for name, data in signal_results.items()},
            "signal_rankings": rankings,
            "fused_ranking": fused_ids,
            "rrf_debug": rrf_debug,
        },
    }


# =========================================================
# PREPARE EXPERIMENTS
# =========================================================
experiment_configs = build_experiment_configs()
print(f"Prepared {len(experiment_configs)} experiment configurations.")
print(f"Parallel experiment workers: {MAX_EXPERIMENT_WORKERS}")
print(f"Weighted RRF enabled: {USE_WEIGHTED_RRF}")

all_experiment_summaries = []
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
    output_id = f"{OUTPUT_LETTER}{paper_id}"

    print("\n==============================")
    print(f"Processing {eval_id}")
    print("==============================")

    ground_truth[eval_id] = correct_model
    save_json_file(
        "8-CRITERIA_SELECTION/user_intent/ground_truth.json",
        ground_truth,
    )

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
            time.sleep(60)
            Eextractor.extract(user_text)
        else:
            print(f"Skipping essential extraction, already exists: {eval_path_e}")

        if not os.path.exists(eval_path_p):
            print(f"Running PreferenceFeaturesExtractor for {sample_key}...")
            time.sleep(60)
            Pextractor.extract(user_text)
        else:
            print(f"Skipping preference extraction, already exists: {eval_path_p}")

        if not os.path.exists(eval_path_q):
            print(f"Running QualityFeaturesExtractor for {sample_key}...")
            time.sleep(60)
            Qextractor.extract(user_text)
        else:
            print(f"Skipping quality extraction, already exists: {eval_path_q}")
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
        lastModified=to_recency_feat(Pfeatures_data.get("lastModified")),
        library_name=to_categorical_feat(Pfeatures_data.get("library_name")),
        likes=to_numeric_feat(Pfeatures_data.get("likes")),
        tensors_total=to_numeric_feat(Pfeatures_data.get("tensors_total")),
        usedStorage=to_numeric_feat(Pfeatures_data.get("usedStorage")),
        datasets=to_categorical_feat(Pfeatures_data.get("datasets")),
        language=to_categorical_feat(Pfeatures_data.get("language")),
        metrics=to_categorical_feat(Pfeatures_data.get("metrics")),
    )

    print("Creating feature bundle")
    Fbundle = FeatureBundle(
        essential=Efeatures,
        preferences=Pfeatures,
        quality=Qfeatures,
        functional=Ffeatures,
    )

    bundle_output_path = os.path.join(
        experiment_output_dir,
        f"{eval_id}_feature_bundle.json",
    )
    save_json_file(bundle_output_path, object_to_dict(Fbundle))

    # =====================================================
    # RUN EXPERIMENTS IN PARALLEL
    # =====================================================
    sample_experiment_results = []

    with ThreadPoolExecutor(max_workers=min(MAX_EXPERIMENT_WORKERS, len(experiment_configs))) as executor:
        future_to_exp = {
            executor.submit(
                run_single_experiment,
                eval_id,
                sample_key,
                correct_model,
                Fbundle,
                exp_cfg,
            ): exp_cfg
            for exp_cfg in experiment_configs
        }

        for future in as_completed(future_to_exp):
            exp_cfg = future_to_exp[future]
            experiment_id = exp_cfg["experiment_id"]
            try:
                result_row = future.result()
            except Exception as exc:
                print(f"[{eval_id}] {experiment_id} failed: {exc}")
                result_row = {
                    "eval_id": eval_id,
                    "sample_key": sample_key,
                    "experiment_id": experiment_id,
                    "correct_model": correct_model,
                    "top1_model": None,
                    "top10_models": [],
                    "correct_rank": None,
                    "hit_at_1": False,
                    "hit_at_10": False,
                    "config": exp_cfg,
                    "mode": "error",
                    "error": str(exc),
                    "artifacts": {},
                }

            sample_experiment_results.append(result_row)
            all_sample_level_results.append({k: v for k, v in result_row.items() if k != "artifacts"})

            full_output_dir = os.path.join(experiment_output_dir, eval_id, experiment_id)
            os.makedirs(full_output_dir, exist_ok=True)

            if result_row.get("mode") == "weighted_rrf":
                artifacts = result_row.get("artifacts", {})
                save_json_file(os.path.join(full_output_dir, "signal_queries.json"), artifacts.get("signal_queries", {}))
                save_json_file(os.path.join(full_output_dir, "signal_feature_groups.json"), artifacts.get("signal_feature_groups", {}))
                save_json_file(os.path.join(full_output_dir, "signal_rankings.json"), artifacts.get("signal_rankings", {}))
                save_json_file(os.path.join(full_output_dir, "fused_ranking.json"), artifacts.get("fused_ranking", []))
                save_json_file(os.path.join(full_output_dir, "rrf_debug.json"), artifacts.get("rrf_debug", {}))
                for signal_name, resp in artifacts.get("signal_responses", {}).items():
                    save_json_file(os.path.join(full_output_dir, f"response_{signal_name}.json"), resp)
            elif result_row.get("mode") == "unified":
                artifacts = result_row.get("artifacts", {})
                save_json_file(os.path.join(full_output_dir, "query.json"), artifacts.get("query", {}))
                save_json_file(os.path.join(full_output_dir, "response.json"), artifacts.get("response", {}))
                save_json_file(os.path.join(full_output_dir, "feature_groups.json"), artifacts.get("feature_groups", {}))

            summary_payload = {k: v for k, v in result_row.items() if k != "artifacts"}
            save_json_file(os.path.join(full_output_dir, "summary.json"), summary_payload)

            try:
                # Use the full feature bundle vs. the expected model file, same as original code.
                builder = ESQueryBuilderAdaptive(
                    mapping=query_mapping,
                    target_hits=exp_cfg["target_hits"],
                    size=exp_cfg["size"],
                    feature_weight_groups=exp_cfg["feature_weight_groups"],
                    boost_config=exp_cfg["boost_config"],
                    enable_rank_functions=ENABLE_RANK_FUNCTIONS,
                    enable_quality_dimensions=ENABLE_QUALITY_DIMENSIONS,
                    enable_feature_locations=ENABLE_FEATURE_LOCATIONS,
                )
                diagnose = builder.compare_fundle_to_sample(
                    features=Fbundle,
                    sample_file=f"8-CRITERIA_SELECTION/test_models/{replace_slash(correct_model)}.json",
                )
                save_json_file(os.path.join(full_output_dir, "diagnose.json"), diagnose)
            except Exception as diag_exc:
                save_json_file(
                    os.path.join(full_output_dir, "diagnose.json"),
                    {"error": str(diag_exc)},
                )

    ranked = sorted(
        sample_experiment_results,
        key=lambda x: (
            999 if x["correct_rank"] is None else x["correct_rank"],
            0 if x["hit_at_1"] else 1,
        ),
    )

    print(f"\nTop experiment summaries for {eval_id}:")
    for row in ranked[:5]:
        print(
            f"{row['experiment_id']} | "
            f"mode={row.get('mode')} | "
            f"rank={row['correct_rank']} | "
            f"hit@1={row['hit_at_1']} | "
            f"top1={row['top1_model']}"
        )

    sample_summary_payload = [{k: v for k, v in row.items() if k != "artifacts"} for row in sample_experiment_results]
    save_json_file(
        os.path.join(experiment_output_dir, f"{eval_id}_all_experiments.json"),
        sample_summary_payload,
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
            "mode": row.get("mode"),
            "n_samples": 0,
            "hit_at_1_count": 0,
            "hit_at_10_count": 0,
            "rank_sum": 0,
            "rank_count": 0,
            "config": row["config"],
            "rrf_weights": row.get("rrf_weights"),
            "rrf_k": row.get("rrf_k"),
        }

    s = summary_by_experiment[exp_id]
    s["n_samples"] += 1
    s["hit_at_1_count"] += int(row["hit_at_1"])
    s["hit_at_10_count"] += int(row["hit_at_10"])

    if row["correct_rank"] is not None:
        s["rank_sum"] += row["correct_rank"]
        s["rank_count"] += 1

final_summary = []
for exp_id, s in summary_by_experiment.items():
    n = max(s["n_samples"], 1)
    avg_rank = None if s["rank_count"] == 0 else round(s["rank_sum"] / s["rank_count"], 4)

    final_summary.append({
        "experiment_id": exp_id,
        "mode": s["mode"],
        "n_samples": s["n_samples"],
        "hit_at_1": round(s["hit_at_1_count"] / n, 4),
        "hit_at_10": round(s["hit_at_10_count"] / n, 4),
        "avg_rank_when_found": avg_rank,
        "config": s["config"],
        "rrf_weights": s.get("rrf_weights"),
        "rrf_k": s.get("rrf_k"),
    })

final_summary = sorted(
    final_summary,
    key=lambda x: (
        -x["hit_at_1"],
        -x["hit_at_10"],
        999999 if x["avg_rank_when_found"] is None else x["avg_rank_when_found"],
    ),
)

save_json_file(
    os.path.join(experiment_output_dir, "global_experiment_summary.json"),
    final_summary,
)

save_json_file(
    os.path.join(experiment_output_dir, "all_sample_level_results.json"),
    all_sample_level_results,
)

print(f"\nNUMBER OF TESTED INPUTS: {non_null_count}")
print("\nBest experiments overall:")
for row in final_summary[:10]:
    print(
        f"{row['experiment_id']} | "
        f"mode={row['mode']} | "
        f"hit@1={row['hit_at_1']} | "
        f"hit@10={row['hit_at_10']} | "
        f"avg_rank={row['avg_rank_when_found']}"
    )
