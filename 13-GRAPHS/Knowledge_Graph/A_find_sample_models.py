from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # pip install pyyaml
except ImportError:
    yaml = None


def load_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON or YAML file into a dict."""
    try:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() in {".yml", ".yaml"}:
            if yaml is None:
                raise ImportError("PyYAML is required for YAML files: pip install pyyaml")
            return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Skipping {path.name}: {e}")
    return None

def name_prefix(name: str) -> str:
    """Return the part before the first '/'."""
    return name.split("/", 1)[0].strip()

def normalize_features(features: Any) -> List[str]:
    """Convert Features into a clean, deduplicated lowercase list."""
    if not isinstance(features, list):
        return []
    cleaned = []
    seen = set()
    for item in features:
        if not isinstance(item, str):
            continue
        value = item.strip().lower()
        if value and value not in seen:
            seen.add(value)
            cleaned.append(value)
    return cleaned


def valid_quality(quality: Any, min_attributes: int = 3) -> bool:
    """
    Quality must be a dict with at least `min_attributes` keys,
    and each value should itself be a dict.
    """
    if not isinstance(quality, dict):
        return False
    subdict_count = sum(1 for v in quality.values() if isinstance(v, dict))
    return subdict_count >= min_attributes


def get_likes(data: Dict[str, Any]) -> int:
    metadata = data.get("Metadata", {})
    if not isinstance(metadata, dict):
        return 0
    likes = metadata.get("likes", 0)
    try:
        return int(likes)
    except (TypeError, ValueError):
        return 0


def feature_overlap_ratio(features_a: List[str], features_b: List[str]) -> float:
    """
    Overlap ratio based on the smaller feature set:
        common_features / min(len(a), len(b))

    This makes '> 20%' mean at least 20% of the smaller model's features overlap.
    """
    set_a = set(features_a)
    set_b = set(features_b)
    if not set_a or not set_b:
        return 0.0
    common = len(set_a & set_b)
    return common / min(len(set_a), len(set_b))


def load_models_from_folder(folder: str) -> List[Dict[str, Any]]:
    folder_path = Path(folder)
    supported_exts = {".json", ".yml", ".yaml"}
    models = []

    for path in folder_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported_exts:
            data = load_file(path)
            if not isinstance(data, dict):
                continue

            name = data.get("modelID")
            features = normalize_features(data.get("Features"))
            quality = data.get("Quality")
            likes = get_likes(data)

            if not isinstance(name, str) or not name.strip():
                continue

            models.append(
                {
                    "name": name.strip(),
                    "Features": features,
                    "Quality": quality,
                    "likes": likes,
                    "source_file": str(path),
                }
            )

    return models


def filter_models(
    models: List[Dict[str, Any]],
    min_features: int = 6,
    max_features: int = 25,
    min_quality_attributes: int = 2,
) -> List[Dict[str, Any]]:
    filtered = []
    for model in models:
        n_features = len(model["Features"])
        if not (min_features <= n_features <= max_features):
            continue
        if not valid_quality(model["Quality"], min_attributes=min_quality_attributes):
            continue
        filtered.append(model)
    return filtered


def find_best_trio(models, min_overlap=0.10):
    best_trio = None
    best_score = -1

    for trio in itertools.combinations(models, 3):
        a, b, c = trio

        # New rule: prefix before "/" must be different for all three
        prefixes = [name_prefix(a["name"]), name_prefix(b["name"]), name_prefix(c["name"])]
        if len(set(prefixes)) < 3:
            continue

        overlaps = [
            feature_overlap_ratio(a["Features"], b["Features"]),
            feature_overlap_ratio(a["Features"], c["Features"]),
            feature_overlap_ratio(b["Features"], c["Features"]),
        ]

        # Looser rule: at least 2 of 3 pairs overlap
        if sum(x > min_overlap for x in overlaps) < 2:
            continue

        score = a["likes"] + b["likes"] + c["likes"]

        if score > best_score:
            best_score = score
            best_trio = trio

    return best_trio


def get_top_3_models(folder: str):
    all_models = load_models_from_folder(folder)
    print("Total models loaded:", len(all_models))

    eligible_models = filter_models(all_models)
    best_trio = find_best_trio(eligible_models)

    if best_trio is None:
        return []

    # sort by likes
    sorted_trio = sorted(best_trio, key=lambda m: m["likes"], reverse=True)
    return sorted_trio


if __name__ == "__main__":
    folder_path = "HF-Models-T7-U"

    models = get_top_3_models(folder_path)

    if models:
        print("Top 3 models:")
        for m in models:
            print(f"{m['name']} (likes: {m['likes']})")
    else:
        print("No trio of models matched all criteria.")