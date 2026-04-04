import os
import json
from collections import defaultdict


# ---------- 1. Collect types + occurrence counts per field path ---------- #

def collect_stats(value, path, type_stats, occurrence_counts):
    """
    Recursively walk the JSON structure and record:
      - Python types of non-null scalar values at each dotted path
      - How many non-null values we've seen for that path (occurrence_counts)
    """
    if isinstance(value, dict):
        for k, v in value.items():
            new_path = f"{path}.{k}" if path else k
            collect_stats(v, new_path, type_stats, occurrence_counts)

    elif isinstance(value, list):
        # For lists, we reuse the same path for each element
        for item in value:
            collect_stats(item, path, type_stats, occurrence_counts)

    else:
        # Scalar (or None)
        if value is not None and path:
            type_stats[path].add(type(value))
            occurrence_counts[path] += 1


def scan_folder_for_stats(folder_path):
    """
    Load all JSON files in folder_path and collect:
      - type_stats: set of Python types seen per path
      - occurrence_counts: non-null scalar occurrence count per path
    """
    type_stats = defaultdict(set)
    occurrence_counts = defaultdict(int)
    file_count = 0

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {filename}: error loading JSON ({e})")
            continue

        file_count += 1

        # Support top-level dict or list of dicts
        if isinstance(data, dict):
            collect_stats(data, "", type_stats, occurrence_counts)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    collect_stats(item, "", type_stats, occurrence_counts)
        else:
            print(f"Skipping {filename}: top-level JSON is {type(data).__name__}, not dict or list")

    if file_count == 0:
        print("No valid JSON files found.")
    else:
        print(f"Scanned {file_count} JSON files.\n")

    return type_stats, occurrence_counts


# ---------- 2. Infer Elasticsearch field types ---------- #

def infer_es_type(py_types):
    """
    Map a set of Python types to an Elasticsearch field type.
    Very simple heuristic — adjust as needed.
    """
    py_types = {t for t in py_types if t is not type(None)}

    if not py_types:
        return None  # no non-null values seen

    if len(py_types) == 1:
        t = next(iter(py_types))
        if t is bool:
            return "keyword" #"boolean" 
        if t is int:
            return "long"
        if t is float:
            return "double"
        if t is str:
            # You can change this to "text" if you prefer
            return "keyword"
        return "keyword"

    # Mixed numeric types -> treat as double
    numeric_types = {int, float}
    if py_types.issubset(numeric_types):
        return "double"

    # Mixed or unknown -> generic string
    return "keyword"


# ---------- 3. Build nested ES mapping structure ---------- #

def insert_field(mapping_properties, path, es_type):
    """
    Given a dotted path like "user.address.city" and an ES type like "keyword",
    insert it into the nested "properties" dict structure.
    """
    parts = path.split(".")
    current = mapping_properties

    for i, part in enumerate(parts):
        is_leaf = (i == len(parts) - 1)

        if is_leaf:
            field_def = current.setdefault(part, {})
            field_def.setdefault("type", es_type)
        else:
            node = current.setdefault(part, {})
            if "properties" not in node:
                node["type"] = "object"
                node["properties"] = {}
            current = node["properties"]


def build_es_mapping(type_stats, occurrence_counts,
                     metadata_prefix="metadata.",
                     metadata_min_occurrences=1000):
    """
    Build an Elasticsearch mapping:
      - Skip fields that never have a non-null value
      - For fields under metadata_prefix (e.g. "metadata."),
        skip if non-null occurrences < metadata_min_occurrences
    """
    root_properties = {}

    for path, py_types in type_stats.items():
        # Have we ever seen a non-null value for this field?
        if occurrence_counts.get(path, 0) == 0:
            # Always null -> skip
            continue

        # Apply metadata filter: e.g. only keep metadata.* fields with >= 1000 occurrences
        if path.startswith(metadata_prefix):
            if occurrence_counts.get(path, 0) < metadata_min_occurrences:
                # Too rare inside metadata -> skip
                continue

        es_type = infer_es_type(py_types)
        if es_type is None:
            continue
        
        # last_segment = path.split(".")[-1]
        # if last_segment.lower() == "lastmodified":
        #     es_type = "date"

        last_segment = path.split(".")[-1]
        if last_segment.lower() == "description":
            es_type = "text"


        insert_field(root_properties, path, es_type)

    return {
        "mappings": {
            "properties": root_properties
        }
    }


# ---------- 4. Collect mapping keys and save them ---------- #

def collect_mapping_keys(mapping, prefix="", out=None):
    """
    Collect all dotted field paths from an ES mapping (including nested).
    """
    if out is None:
        out = []

    props = mapping.get("properties", {})
    for key, value in props.items():
        path = f"{prefix}.{key}" if prefix else key
        out.append(path)

        if isinstance(value, dict) and "properties" in value:
            collect_mapping_keys(value, path, out)

    return out


if __name__ == "__main__":
    # 👉 Change this to your folder path
    VERSION = "T7"
    folder = f"HF-Models-{VERSION}"

    # Scan folder and gather stats
    type_stats, occurrence_counts = scan_folder_for_stats(folder)

    # Build mapping with metadata filtering (< 1000 occurrences removed under "metadata")
    es_mapping = build_es_mapping(
        type_stats,
        occurrence_counts,
        metadata_prefix="metadata.",        # root key "metadata"
        metadata_min_occurrences=1000       # threshold
    )

    # Save mapping to file
    mapping_file = f"8-CRITERIA_SELECTION/es_mapping_{VERSION}.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(es_mapping, f, indent=2, ensure_ascii=False)
    print(f"Saved mapping to {mapping_file}")

    # Collect and save all field keys from the mapping
    all_keys = collect_mapping_keys(es_mapping["mappings"])
    all_keys = sorted(all_keys)

    keys_file = f"8-CRITERIA_SELECTION/mapping_keys_{VERSION}.json"
    with open(keys_file, "w", encoding="utf-8") as f:
        json.dump(all_keys, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_keys)} keys to {keys_file}")


