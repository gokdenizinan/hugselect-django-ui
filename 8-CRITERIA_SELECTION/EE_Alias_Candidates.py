import json
import os
from typing import Any, List, Set
from E_utils import save_to_json

def get_nested_value(data: dict, key_path: List[str]) -> Any:
    """
    Safely retrieves a nested value from a dictionary using a list of keys.
    """
    for key in key_path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def collect_feature_values(folder_path: str, feature: str) -> List[Any]:
    """
    Collects all unique values for a given feature across all JSON files in a folder.

    Args:
        folder_path: Path to folder containing JSON files
        feature: Feature name (e.g. "author", "Metadata.license", "Quality.Security.score")

    Returns:
        Sorted list of unique values
    """
    values: Set[Any] = set()
    key_path = feature.split(".")

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            value = get_nested_value(data, key_path)

            if value is None:
                continue

            # Handle lists (e.g. tags, datasets)
            if isinstance(value, list):
                values.update(value)
            else:
                values.add(value)

        except (json.JSONDecodeError, OSError):
            # Skip unreadable or malformed files
            continue

    return sorted(values)

# ====================
# ====================
# ====================
# EXTRA FOR CREATING A LIST OF FEATURES WITH CORRECPONING AVAILABLE VALUES
# ====================
# ====================
# ====================
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict


def _freeze(value: Any) -> Any:
    """
    Convert JSON-like values into a hashable canonical form.
    """
    if isinstance(value, dict):
        return tuple(
            (k, _freeze(v)) for k, v in sorted(value.items(), key=lambda kv: kv[0])
        )

    if isinstance(value, list):
        frozen = [_freeze(v) for v in value if v is not None]
        return tuple(sorted(frozen, key=lambda x: repr(x)))

    return (type(value).__name__, value)


def _is_scalar(x: Any) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))


def flatten_json(data: Any, parent_key: str = "") -> Dict[str, Any]:
    items: Dict[str, Any] = {}

    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            items.update(flatten_json(v, new_key))
        return items

    if isinstance(data, list):
        if parent_key == "":
            parent_key = "__root__"

        if all(_is_scalar(x) for x in data):
            items[parent_key] = data
            return items

        for i, elem in enumerate(data):
            idx_key = f"{parent_key}[{i}]"
            items.update(flatten_json(elem, idx_key))
        return items

    items[parent_key] = data
    return items


def analyze_features(folder_path: str, sample_size: int = 5, seed: int = 42):
    random.seed(seed)

    assigned_counts = defaultdict(int)
    unique_values = defaultdict(set)

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        flat = flatten_json(data)
        seen_in_file = set()

        for feature, value in flat.items():
            if value is None:
                continue

            frozen_value = _freeze(value)

            if feature not in seen_in_file:
                assigned_counts[feature] += 1
                seen_in_file.add(feature)

            unique_values[feature].add(frozen_value)

    # Build result with random samples
    result = {}

    for feature, values in unique_values.items():
        values_list = list(values)

        if len(values_list) > sample_size:
            samples = random.sample(values_list, sample_size)
        else:
            samples = values_list

        result[feature] = {
            "assigned_count": assigned_counts[feature],
            "unique_count": len(values_list),
            "sample_values": samples,
        }

    return result

def sorted_features_by_assigned_count(stats: dict, threshold: int = 1000):
    return sorted(
        (
            (key, v["assigned_count"], [str(sample)[:100] for sample in v.get("sample_values", [])])
            for key, v in stats.items()
            if v.get("assigned_count", 0) > threshold
        ),
        key=lambda x: x[1],
        reverse=True
    )

def remove_prefixed_items(
    input_path: str,
    output_path: str,
    prefixes=None,
    case_insensitive=False,
):
    """
    Load a JSON list of strings, remove items starting with given prefixes,
    and write the result to a new JSON file.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if case_insensitive:
        prefixes = tuple(p.lower() for p in prefixes)
        filtered = [
            s for s in items
            if isinstance(s, str) and not s.lower().startswith(prefixes)
        ]
    else:
        filtered = [
            s for s in items
            if isinstance(s, str) and not s.startswith(prefixes)
        ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)

    return len(items), len(filtered)


if __name__ == "__main__":
    # SADECE FEATURELA YENIDEN RUNLA ALIAS LISTESI YARATMAK ICIN
    folder_path = "HF-Models-T7"
    save_path = "8-CRITERIA_SELECTION/alias_candidates/"
#     meta_features = [
#     "basemodels",
#     "datasets",
#     "license",
#     "model_type",
#     "library_name",
#     "pipeline_tag",
#     "tags",
#     "language",
#     "metrics",
# ]
#     features = ["author",
#                 "Features",
# ]

    meta_features = [
    "gated",
    "language",
]
    features = [
        "Features",
]

    for feature_name in meta_features:
        values = collect_feature_values(folder_path, f"Metadata.{feature_name}")
        save_to_json(values, save_path + feature_name)

    for feature_name in features:
        values = collect_feature_values(folder_path, f"{feature_name}")
        save_to_json(values, save_path + feature_name)

    # # REMOVE ITEMS FROM CANDIDATES BECAUSE THEY ARE TOO BIG
    # tags_input_path = "8-CRITERIA_SELECTION/alias_candidates/tags.json"
    # tags_output_path = "8-CRITERIA_SELECTION/alias_candidates/tags_2.json"
    # before, after = remove_prefixed_items(tags_input_path, tags_output_path, prefixes= ("base_model", "dataset", "arxiv"), case_insensitive=True)

    # print(f"Removed {before - after} items")

    # # STATS OF FEATURES
    # stats = analyze_features(folder_path)
    # stats_save_path = "8-CRITERIA_SELECTION/STATS_FEATURE_T6"
    # save_to_json(stats, stats_save_path)
    # print("done")


    # with open("8-CRITERIA_SELECTION/STATS_FEATURE_T6.json", "r", encoding="utf-8") as f:
    #     stats_feature = json.load(f)
    # keys = sorted_features_by_assigned_count(stats_feature, threshold=1100)
    # for i in keys:
    #     print("")
    #     print(i[0])
    #     print(i[1])
    #     print(i[2])
