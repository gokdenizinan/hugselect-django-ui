import json
import os
from typing import Any, Dict, List, Optional
from dateutil import parser


def normalize_metrics(metrics):
    """
    - If metrics is a list: keep it.
    - Otherwise (dict, str, number, None, etc.): return None.
    """
    if isinstance(metrics, list):
        return metrics
    return None


def get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safely get a nested value using a dotted path, e.g. 'metadata.cardData.license'.
    Works with dicts and lists (numeric segments for list indices).
    """
    if d is None:
        return default

    current = d
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part, default)
        elif isinstance(current, list):
            # allow numeric index for lists
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return default
        else:
            return default

        if current is None:
            return default

    return current


QUALITY_KEYS = [
    "Functional Suitability",
    "Reliability",
    "Compatibility",
    "Flexibility",
    "Interaction Capability",
    "Maintainability",
    "Performance Efficiency",
    "Security",
]


def extract_quality(raw: dict) -> dict:
    """
    Build the Quality dict from:
      1) top-level keys (preferred)
      2) fallback 'Quality' / 'quality' nested dict
    Always returns a dict with all QUALITY_KEYS present (missing -> None).
    """
    quality: dict = {}

    # 1) Top-level keys: e.g. raw["Functional Sustainability"], raw["Security"], ...
    for key in QUALITY_KEYS:
        if key in raw:
            quality[key] = raw[key]

    # 2) If not present at top level, look into nested 'Quality' / 'quality'
    quality_container = raw.get("Quality") or raw.get("quality") or {}
    if isinstance(quality_container, dict):
        for key in QUALITY_KEYS:
            if key not in quality and key in quality_container:
                quality[key] = quality_container[key]

    # 3) Guarantee all keys exist; fill missing with None
    for key in QUALITY_KEYS:
        quality.setdefault(key, None)

    return quality



def transform_model_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform one HF model dictionary into the desired compact format.
    """
    metadata = raw.get("metadata", {}) or {}

    # --- modelID, author, description, Features -----------------------------
    # Try multiple possible locations for robustness.
    model_id = (
        raw.get("modelID")
        or raw.get("modelId")
        or raw.get("id")
        or get_nested(metadata, "id")
    )

    author = (
        raw.get("author")
        or get_nested(metadata, "author")
        or get_nested(metadata, "cardData.author")
        or get_nested(metadata, "cardData.owner")
    )

    description = (
        raw.get("description")
        or get_nested(metadata, "description")
        or get_nested(metadata, "cardData.description")
        or get_nested(metadata, "cardData.model_description")
    )

    # Features: assume list of strings, but normalize different shapes
    features_raw = (
        raw.get("Features")
        or raw.get("features")
        or get_nested(metadata, "cardData.features")
        or []
    )
    if isinstance(features_raw, str):
        Features: List[str] = [features_raw]
    elif isinstance(features_raw, list):
        Features = [str(x) for x in features_raw]
    else:
        Features = []

    # --- Quality ------------------------------------------------------------
    # If you already have a "Quality" or "quality" dict, pick those keys from it
    # --- Quality ------------------------------------------------------------
    # --- Quality ------------------------------------------------------------
    Quality = extract_quality(raw)

    # --- Metadata -----------------------------------------------------------
    # # basemodels may be a list of objects; collect their "_id"
    # base_models = get_nested(metadata, "baseModels.models") or []
    # basemodels: Optional[List[Any]]
    # if isinstance(base_models, list):
    #     basemodels = [
    #         m.get("_id")
    #         for m in base_models
    #         if isinstance(m, dict) and "_id" in m
    #     ]
    # else:
    #     basemodels = None

    # disabled : yok cunku bunlarin toptan cikarilmasi lazim datasetten aq
    # childrenmodelcount yok cunku onun ben aminakoyin bi boka yaramiyor 
    # transformersinfo.auto_model: bu ne anlamadim
    # config.model_type: bu ne tam olarak
    # config.architectures: nu ne
    # transformersinfo.processor: bu ne
    #carddata inference: cokta onemli degil gibi geldi bana
    # gguff.total storage size ikle ayni bilgiyi veriyor
    # cardData.model-index.results.task.type", spaces, private gitti sildim boslar

    Metadata = {
        "pipeline_tag": get_nested(metadata, "pipeline_tag"), # in # in transformersinfo.pipeline_tag # in carddata.pipeline_tag (67770 tanede var ne bok yiycez) (must olarak ariyoruz bunu sonucta) (pipeline tagi yoksa direkt tagden biseyi pipeline tage gecirelim)
        "tags": get_nested(metadata, "tags"), # in # in carddata
        "license": get_nested(metadata, "license"), # in # incarddata
        "model_type": get_nested(metadata, "config.model_type"), # in # carddata.model_type tada var (agabuga tuhaf tuhaf isimleri var halletmenin yolunu bul)
        "basemodels": get_nested(metadata, "cardData.base_model"), #basemodels, # in
        "datasets": get_nested(metadata, "cardData.datasets"), # in
        "library_name": get_nested(metadata, "library_name"), # in #in carddata.library_name
        "likes": get_nested(metadata, "likes"), # in
        "downloads_all_time": get_nested(metadata, "downloads_all_time"), # in
        "downloads_last_30_days": get_nested(metadata, "downloads_last_30_days"), # in
        "file_count": get_nested(metadata, "file_count"), # in
        "gated": str(get_nested(metadata, "gated")), # in
        "url": get_nested(metadata, "url"), # in
        "tensors_total": get_nested(metadata, "tensors.total"), # in
        "usedStorage": get_nested(metadata, "usedStorage"), # in
        "language": get_nested(metadata, "cardData.language"), # added (sq, grc, chm bundlar nasil temsil ediliyor tam olarak)
    }

    mm = get_nested(metadata, "metrics")
    Metadata["metrics"] = (mm if isinstance(mm, list) and all(isinstance(x, str) for x in mm) else None)

    lm = get_nested(metadata, "lastModified")
    Metadata["lastModified"] = str(parser.parse(lm) if lm else None) # in # in last_modified
    return {
        "modelID": model_id, # in
        "author": author, # in
        "description": description, # in
        "Features": Features, # in
        "Quality": Quality, # in
        "Metadata": Metadata, # in
    }

def transform_folder(input_dir: str, output_dir: str) -> None:
    """
    Read all .json files from input_dir, transform them, and write new JSON
    to output_dir with the same file names.
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".json"):
            continue

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        with open(in_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # If files sometimes contain a list of dicts, handle that too
        if isinstance(raw, list):
            transformed = [transform_model_dict(item) for item in raw]
        else:
            transformed = transform_model_dict(raw)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transformed, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    transform_folder("HF-Models-T5", "HF-Models-T6")
