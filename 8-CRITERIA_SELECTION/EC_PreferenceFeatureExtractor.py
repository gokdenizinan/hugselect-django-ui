from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from EB_LLM_Client import LLMClient, LLMResponse, LoggingLLMClient
import json
from E_utils import parse_llm_json_flex  # adjust import
from EA_Features import (PreferenceFeatures, CategoricalFeature, NumericFeature, BoolFeature, RecencyFeature)
from EA_Features import (get_llm_text, to_categorical_feat, to_numeric_feat, to_bool_feat, to_recency_feat)

# === LLM CLIENT ===
class PreferenceFeaturesExtractor:
    def __init__(self, llm_client):
        self.llm = llm_client

    def extract(self, user_text: str) -> PreferenceFeatures:
        """
        Extract essential features from user input using LLM.
        Expect JSON structured output.
        """
        prompt = f"""
You will extract *preference and constraint information* about models from the user's message.

USER MESSAGE:
{user_text}

You must return a single JSON object with the following top-level fields:

PreferenceFeatures:
- basemodels: CategoricalFeature or null
- license_name: CategoricalFeature or null
- downloads_all_time: NumericFeature or null
- downloads_last_30_days: NumericFeature or null
- file_count: NumericFeature or null
- gated: BoolFeature or null
- lastModified: RecencyFeature or null
- library_name: CategoricalFeature or null
- likes: NumericFeature or null
- tensors_total: NumericFeature or null
- usedStorage: NumericFeature or null
- datasets: CategoricalFeature or null
- language: CategoricalFeature or null
- metrics: CategoricalFeature or null

Where the nested types are:

NumericFeature:
- op: one of "gte", "lte", "gt", "lt", "eq", "approx", or null
- value: integer or null
- value_direction: one of "very_low", "low", "medium", "high", "very_high", or null
  (Use value_direction only when no explicit numeric value is given but user indicates preference for high/low quantity.)
- priority: one of "must", "strong_prefer", "prefer", "avoid", or null

BoolFeature:
- value: true, false, or null
- priority: one of "must", "strong_prefer", "prefer", "avoid", or null

CategoricalFeature:
- include: list of strings (values the user wants)
- exclude: list of strings (values the user rejects)
- priority: one of "must", "strong_prefer", "prefer", "avoid", or null

RecencyFeature:
- max_age_days: integer or null
- value_direction: one of "very_low", "low", "medium", "high", "very_high", or null
  (Use value_direction to indicate strength of preference for recency when no explicit time window is given.
   higher direction = stronger preference for newer models.)
- priority: one of "must", "strong_prefer", "prefer", "avoid", or null

INTERPRETATION RULES:

- Only encode preferences explicitly stated or very strongly implied.
- If there is no information for a top-level field, set it to null.
- For CategoricalFeature, include and exclude must always be present as lists.
- Use JSON booleans true/false and JSON null.

Numeric operator mapping:
- "at least", "no less than", ">= X" → op="gte", value=X
- "at most", "no more than", "<= X" → op="lte", value=X
- "more than", "above", "> X" → op="gt", value=X
- "less than", "under", "< X" → op="lt", value=X
- "exactly", "= X", "equal to" → op="eq", value=X
- "around", "roughly", "~X" → op="approx", value=X

Priority mapping hints:
- "must", "required", "absolutely need" → must
- "really want", "very important" → strong_prefer
- "prefer", "would like", "nice to have" → prefer
- "avoid", "do not want", "no X" → avoid
- If implied but weak → prefer
- If unclear → null

Recency hints:
- "latest", "very new", "newest only" → value_direction="very_high"
- "not older than N days/months/years" → compute max_age_days
- "recent is better but not required" → value_direction="medium"
- If irrelevant → lastModified = null

Gated / private rules:
- "I don't want gated models", "avoid gated" → gated.value=false, priority based on strength
- "must be gated" → gated.value=true
- "must be private" / "no public models" → private.value=true
- "must be public" / "not private" → private.value=false

Library / basemodel examples:
- "Transformers", "Diffusers", "PyTorch", "TensorFlow", "Keras", "gguf", "llama" → library_name or basemodels include entries
- "no TensorFlow" → library_name.exclude includes "tensorflow"

License examples:
- "Apache-2.0", "MIT", "permissive", "non-commercial", "OpenRAIL" → license_name include/exclude
- "no non-commercial" → license_name.exclude includes relevant terms

Datasets:
- "trained on X" → datasets.include includes X
- "avoid models trained on Y" → datasets.exclude includes Y

GENERAL REQUIREMENTS:
- Return ONLY valid JSON.
- Do not include comments, explanations, or trailing text.
- All top-level fields must be present.
- Fields with no information must be null.
- CategoricalFeature.include and .exclude must always be lists.

Example output format (structure only):

{{
  "basemodels": {{
    "include": ["llama-3", "mistral"],
    "exclude": [],
    "priority": "prefer"
  }},
  "license_name": null,
  "downloads_all_time": {{
    "op": "gte",
    "value": 10000,
    "value_direction": "high",
    "priority": "prefer"
  }},
  "downloads_last_30_days": null,
  "file_count": null,
  "gated": {{
    "value": false,
    "priority": "strong_prefer"
  }},
  "lastModified": {{
    "max_age_days": 90,
    "value_direction": "high",
    "priority": "must"
  }},
  "library_name": {{
    "include": ["pytorch"],
    "exclude": ["tensorflow"],
    "priority": "must"
  }},
  "likes": null,
  "tensors_total": null,
  "usedStorage": null,
  "datasets": {{
    "include": [],
    "exclude": ["c4"],
    "priority": "avoid"
  }},
  "language": {{
    "include": ["English"],
    "exclude": [],
    "priority": "must"
  }},
  "metrics": {{
    "include": ["accuracy"],
    "exclude": [],
    "priority": "prefer"
  }},
}}
"""


        # raw_text = self.llm.generate(prompt).text
        llm_result = self.llm.generate(prompt)
        raw_text = get_llm_text(llm_result)

        # 🔹 Use the flexible parser instead of raw json.loads
        try:
            data = parse_llm_json_flex(raw_text)
        except ValueError as e:
            print("---- RAW LLM OUTPUT ----")
            print(raw_text)
            print("------------------------")
            raise ValueError(f"Couldn't parse JSON: {data}")


        
        prefs = PreferenceFeatures(
            basemodels=to_categorical_feat(data.get("basemodels")),
            license_name=to_categorical_feat(data.get("license_name")),
            downloads_all_time=to_numeric_feat(data.get("downloads_all_time")),
            downloads_last_30_days=to_numeric_feat(data.get("downloads_last_30_days")),
            file_count=to_numeric_feat(data.get("file_count")),
            gated=to_bool_feat(data.get("gated")),
            lastModified=to_recency_feat(data.get("lastModified")),
            library_name=to_categorical_feat(data.get("library_name")),
            likes=to_numeric_feat(data.get("likes")),
            tensors_total=to_numeric_feat(data.get("tensors_total")),
            usedStorage=to_numeric_feat(data.get("usedStorage")),
            datasets=to_categorical_feat(data.get("datasets")),
            language=to_categorical_feat(data.get("language")),
            metrics=to_categorical_feat(data.get("metrics")),
        )
    
        return prefs

if __name__ == "__main__":

    GEMINI_API_KEY = ""
    llm_client = LLMClient(
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.5-flash",
        max_retries=5,
        retry_delay_seconds=20.0,
    )

    # wrap it with a logger
    logger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir="8-CRITERIA_SELECTION/user_intent/preference_features",
        print_output=True,
        save_file= "deneme_preference_1.json",   # optional
    )

    extractor = PreferenceFeaturesExtractor(logger)

    user_text = "Machine translation foundational models supporting low-resource languages. the model should be gated, and i want a model that is recently published and that is highly popular and widely used."

    # features = extractor.extract(user_text)


    #=== ELASTIC SEARCH CLIENT === 
    print("importing creating feature bundle...")

    import json
    import glob

    files = glob.glob("8-CRITERIA_SELECTION/user_intent/preference_features/deneme_preference_1.json")

    for fpath in files:
        with open(fpath, "r") as f:
            raw = json.load(f)
        # the actual model output
        raw_text = get_llm_text(raw[0])  # raw[0] cunku birden fazla llm outputu cikma ihitmaline karsi liste olarak kaydediyoruz outputlari
        features = parse_llm_json_flex(raw_text)
        print(raw_text)
        print("=")

        # print(raw_text["task"])
        print("=========")
        
        print(features)
        print("=")
        print(features["usedStorage"])
