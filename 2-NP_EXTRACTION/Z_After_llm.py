import json
from typing import List, Any, Dict
import re

with open("10-EVALUATION/llm_nps/output_A0.json", "r", encoding="utf-8") as f:
    after = json.load(f)

with open("2-NP_EXTRACTION/NP_comfy_PP.json", "r", encoding="utf-8") as f:
    np_info = json.load(f)

def parse_llm_json_flex(text: str) -> List[str]:
    s = text.strip()

    parsed: Dict[str, Any] | None = None

    # 1) Try plain JSON directly
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2) Try to extract from ```json ... ``` or ``` ... ```
    if parsed is None:
        fence_match = re.search(
            r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE
        )
        if fence_match:
            inner = fence_match.group(1).strip()
            try:
                parsed = json.loads(inner)
            except json.JSONDecodeError:
                pass

    # 3) Try substring between first '{' and last '}'
    if parsed is None:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = s[start : end + 1]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # 4) Give up if nothing parsed
    if parsed is None:
        snippet = s[:300].replace("\n", "\\n")
        raise ValueError(f"Could not parse JSON from LLM output. Snippet: {snippet}")

    # 5) Normalize output → List[str]
    if isinstance(parsed, dict):
        # Common case: {"features": [...]}
        for value in parsed.values():
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                return value

    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
        return parsed

    raise ValueError(f"Parsed JSON but could not extract List[str]: {parsed}")

import json
import re

def extract_json_from_output(output_str):
    # Remove code fences if present
    cleaned = re.sub(r"```.*?\n|```", "", output_str, flags=re.DOTALL).strip()
    # Extract the JSON object
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    return json.loads(match.group())

unified_dict = {}

for item in after.values():
    output = item.get("output", "")
    parsed = extract_json_from_output(output)
    if parsed:
        unified_dict.update(parsed)

with open("2-NP_EXTRACTION/NP_PP_keep.json", "w", encoding="utf-8") as f:
    json.dump(unified_dict, f, indent=2, ensure_ascii=False)