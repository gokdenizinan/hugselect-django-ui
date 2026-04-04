from dataclasses import dataclass, field
from typing import List, Optional, Iterable, Set
from typing import List, Dict, Optional, Literal
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from Ab_generic_terms import GENERIC_TERMS, EXCLUDE, WEAK_HEADS
from EC_FunctionalFeatureExtractor import FunctionalFeatures

# === FEATURE TYPES ===

# Strength of the preference, used across all fields
PreferencePriority = Literal["must", "strong_prefer", "prefer", "avoid"]
# Operators for numeric constraints
NumericOp = Literal["gte", "lte", "gt", "lt", "eq", "approx"]

@dataclass
class NumericFeature:
    """
    Encodes a numeric preference like:
    - "at least 100 likes"
    - "small model size"
    - "very popular model"

    Interpretation:
      - If `value` is not None, use (op, value) as the explicit constraint.
      - If `bucket` is set, interpret it in your own code (e.g. "high" => top 10%).
      - If both are None, this field expresses no numeric preference.
    """
    op: Optional[NumericOp] = None          # e.g. "gte", "lt", "approx"
    value: Optional[int] = None             # explicit threshold when known
    bucket: Optional[str] = None            # e.g. "low", "medium", "high", "small", "large"
    priority: Optional[PreferencePriority] = None  # "must"/"prefer"/"avoid"

@dataclass
class BoolFeature:
    """
    Encodes a boolean-like preference, e.g.:
      - 'no gated models'    -> value=False, priority="must"
      - 'prefer public'      -> value=False, priority="strong_prefer"
      - 'Space demo is nice' -> spaces.value=True, priority="prefer"
    """
    value: Optional[str] = None                    # True/False if they care, None if indifferent
    priority: Optional[PreferencePriority] = None   # "must", "strong_prefer", "prefer", "avoid"

@dataclass
class CategoricalFeature:
    """
    Encodes preferences over categorical/string fields:
      - license_name ("mit", "apache-2.0", or bucket 'permissive')
      - library_name ("transformers", bucket 'pytorch_transformers')
      - basemodels (["llama-3", "mistral"])
      - datasets (include/exclude certain datasets)

    `bucket` is for vague descriptions like "permissive", "open source", etc.,
    which your code later expands into concrete sets.
    """
    include: List[str] = field(default_factory=list)    # explicit values user wants
    exclude: List[str] = field(default_factory=list)    # explicit values user rejects
    bucket: Optional[str] = None                        # semantic label, e.g. "permissive", "llama", "pytorch"
    priority: Optional[PreferencePriority] = None       # importance of this preference

@dataclass
class RecencyFeature:
    """
    Preferences about how recent the model should be.

    - `max_age_days`: explicit numeric upper bound (e.g., "not older than 90 days")
    - `bucket`: vague recency category your code maps to days
        e.g. "very_recent", "recent", "normal", "old"
    """
    max_age_days: Optional[int] = None               # e.g. 90 for "not older than 3 months"
    bucket: Optional[str] = None                     # "very_recent", "recent", "normal", "old"
    priority: Optional[PreferencePriority] = None    # "must", "strong_prefer", "prefer", "avoid"

# BUNU GELISTIRMEN LAZIM = HER SEY STANDARD OLSUN DIYE, ANLAMASI DAHA KOLA OLIYO
@dataclass
class Preference:
    type: Literal["numeric", "boolean", "categorical", "recency"]
    weight: PreferencePriority
    data: dict

# === FEATURE CATEGORIES ===
@dataclass
class EssentialFeatures: # cok onemli oldugu icin preferencedan ayri
    task: str               # "text-classification", "image-generation", etc. # pipeline tage
    domain: Optional[str]   # "medical", "code", "finance", etc.
    model_name: Optional[str] # modeli ismiyle ararsa napicaz cakal
    author: Optional[str]   # author
    objective: Optional[str] # Quality, Popularity (birkac tane) emin degilim lazim mi # to decide wheter a feature is a constrait or not.

    task_aliases: List[str] = field(default_factory=list) # bunlarla model_type, pipeline_tag, tags, functional features. falan arancak
    domain_aliases: List[str] = field(default_factory=list)

@dataclass
class PreferenceFeatures:
    """
    Top-level container for all extracted model preferences.

    IMPORTANT:
    - If a field is None, treat it as "no preference expressed".
    - If a field is not None but its internals are all default/empty,
      you can also treat it as "effectively no preference".
    """
    
    basemodels: Optional[CategoricalFeature] = None # CAT
    license_name: Optional[CategoricalFeature] = None # CAT
    downloads_all_time: Optional[NumericFeature] = None # NUM
    downloads_last_30_days: Optional[NumericFeature] = None # NUM
    file_count: Optional[NumericFeature] = None # NUM
    gated: Optional[BoolFeature] = None # CAT
    lastModified: Optional[RecencyFeature] = None # NUM
    library_name: Optional[CategoricalFeature] = None # CAT
    likes: Optional[NumericFeature] = None # NUM
    tensors_total: Optional[NumericFeature] = None # NUM
    usedStorage: Optional[NumericFeature] = None # NUM
    datasets: Optional[CategoricalFeature] = None # CAT
    language: Optional[str] = None # CAT
    metrics: Optional[List[str]] = None # CAT  

@dataclass
class QualityFeatures:
    Functional_Suitability: Optional[int]
    Compatibility: Optional[int]
    Performance_Efficiency: Optional[int]
    Reliability: Optional[int]
    Interaction_Capability: Optional[int]
    Security: Optional[int]
    Maintainability: Optional[int]
    Flexibility: Optional[int]

# === OTHER OBJECTS ===

@dataclass
class FeatureBundle:
    essential: EssentialFeatures
    preferences: PreferenceFeatures
    functional: FunctionalFeatures
    quality: QualityFeatures

@dataclass
class ModelResult:
    model_id: str
    score: float
    metadata: Dict

@dataclass
class UserQuery:
    raw_text: str
    user_id: Optional[str] = None

# === USEFUL FUNCTIONS ===

def get_llm_text(result):
    if hasattr(result, "text"):
        return result.text
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    raise TypeError(f"Don't know how to extract text from type: {type(result)}")

def to_numeric_feat(obj):
    if obj is None:
        return None
    return NumericFeature(
        op=obj.get("op"),
        value=obj.get("value"),
        bucket=obj.get("bucket"),
        priority=obj.get("priority"),
    )


# def to_bool_feat(obj):
#     if obj is None:
#         return None
#     return BoolFeature(
#         value=obj.get("value"),
#         priority=obj.get("priority"),
#     )

def to_bool_feat(obj):
    if obj is None:
        return None

    value = obj.get("value")
    if isinstance(value, str):
        value = value.upper()

    return BoolFeature(
        value=value,
        priority=obj.get("priority"),
    )


def to_categorical_feat(obj):
    if obj is None:
        return None

    # ensure lists exist even if empty
    include = obj.get("include") or []
    exclude = obj.get("exclude") or []

    return CategoricalFeature(
        include=list(include),
        exclude=list(exclude),
        bucket=obj.get("bucket"),
        priority=obj.get("priority"),
    )


def to_recency_feat(year, *, priority="must"):
    if year is None:
        return None

    # normalize to string
    year_str = str(year).strip()

    # basic validation: must be 4-digit year
    if not year_str.isdigit() or len(year_str) != 4:
        raise ValueError(f"Invalid year format: {year}")

    return CategoricalFeature(
        include=[year_str],
        exclude=[],
        bucket=None,
        priority=priority,
    )

# def to_recency_feat(obj):
#     if obj is None:
#         return None
#     return RecencyFeature(
#         max_age_days=obj.get("max_age_days"),
#         bucket=obj.get("bucket"),
#         priority=obj.get("priority"),
#     )

