import json
import re
from typing import Any, Dict
import json
from dataclasses import is_dataclass, asdict
from enum import Enum

#LLM json dictionary olarak output veriyor ya. ama her zaman belirli bir standarda uymayabiliyor. onu garantilemek icin bu funciton
def parse_llm_json_flex(text: str) -> Dict[str, Any]:
    """
    Try to parse JSON from an LLM response that may be:
    - raw JSON
    - JSON inside ``` or ```json fences
    - JSON with some extra explanation around it

    Raises ValueError if it cannot find valid JSON.
    """
    s = text.strip()

    # 1) Try plain JSON directly
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2) Try to extract from ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    # 3) Try substring between first '{' and last '}' (common case)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 4) Give up
    snippet = s[:300].replace("\n", "\\n")
    raise ValueError(f"Could not parse JSON from LLM output. Snippet: {snippet}")

import json
import os


def _json_default(o):
    # dataclasses (Stage, ClauseSpec, and any nested dataclasses)
    if is_dataclass(o):
        return asdict(o)

    # Enums (likely PreferencePriority)
    if isinstance(o, Enum):
        return o.value

    # If something else sneaks in, make the error explicit
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def save_to_json(data, base_path, index = ""):
    """
    Save text or dictionary as a JSON file named with an index.

    :param data: str or dict
    :param index: int or str
    :param base_path: directory path where file will be saved
    """
    # os.makedirs(base_path, exist_ok=True)

    file_path = os.path.join(base_path +f"{index}.json")

    # Wrap text into a dict so JSON is always valid
    if isinstance(data, str):
        data = {"text": data}

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2,default=_json_default )

    return file_path

def object_to_dict(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif isinstance(obj, list):
        return [object_to_dict(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {k: object_to_dict(v) for k, v in obj.__dict__.items()}
    else:
        return str(obj)
