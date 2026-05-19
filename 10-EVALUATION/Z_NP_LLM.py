import os
import json
import time
import copy
import requests
from typing import Dict, Any, Optional


class LlamaClient:
    """
    LLaMA (Together API) client with:
      - prompt() method
      - automatic JSON output saving
      - checkpoint persistence
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        output_path: str,
        checkpoint_path: str,
        url: str = "https://api.together.xyz/v1/chat/completions",
        max_retries: int = 5,
        timeout: int = 30,
        sleep_on_success: float = 1.0,
    ):
        self.api_key = api_key
        self.model = model
        self.url = url
        self.max_retries = max_retries
        self.timeout = timeout
        self.sleep_on_success = sleep_on_success

        self.output_path = output_path
        self.checkpoint_path = checkpoint_path

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.results: Dict[str, Dict[str, Any]] = self._load_json(self.output_path, {})
        self.checkpoint: Dict[str, Any] = self._load_json(
            self.checkpoint_path, {"last_key": -1}
        )

    # ---------------- PUBLIC API ---------------- #

    def prompt(
        self,
        prompt_text: str,
        *,
        key: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sends a prompt to LLaMA and persists the result.
        """
        meta = meta or {}

        output_text = self._call_llama(prompt_text)

        record = {
            **copy.deepcopy(meta),
            "prompt": prompt_text,
            "output": output_text,
            "model_used": self.model,
        }

        if key is None:
            key = str(len(self.results))

        self.results[str(key)] = record

        self._save_json(self.output_path, self.results)
        self.checkpoint["last_key"] = key
        self._save_json(self.checkpoint_path, self.checkpoint)

        if self.sleep_on_success:
            time.sleep(self.sleep_on_success)

        return record

    def last_checkpoint(self) -> int:
        return int(self.checkpoint.get("last_key", -1))

    # ---------------- INTERNALS ---------------- #

    def _call_llama(self, prompt_text: str) -> str:
        data = {
            "messages": [{"role": "user", "content": prompt_text}],
            "model": self.model,
        }

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=data,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"].strip()

                # non-200
                last_error = {
                    "status_code": response.status_code,
                    "body": response.text,
                }
                time.sleep(min(2 ** attempt, 30))

            except requests.exceptions.RequestException as e:
                last_error = {"exception": str(e)}
                time.sleep(min(2 ** attempt, 30))

        raise RuntimeError(
            f"LLaMA request failed after {self.max_retries} retries: {last_error}"
        )

    # ---------------- IO ---------------- #

    @staticmethod
    def _load_json(path: str, default: Any):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    @staticmethod
    def _save_json(path: str, obj: Any):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

# --------------- USAGE EXAMPLE --------------- #
# BUILD PROMPT

import json
from pathlib import Path
from typing import List, Dict, Any, Union


def _chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

from typing import Dict, Any, List, Tuple

try:
    import tiktoken
except ImportError:
    tiktoken = None


def top_10_longest_prompts_by_tokens(
    data: Dict[str, Dict[str, Any]],
    model_name: str = "gpt-4o-mini"
) -> List[Tuple[str, int]]:
    """
    Counts tokens in data[batch_key]["prompt"] and returns
    the top 10 longest as (batch_key, token_count).
    """

    if tiktoken is None:
        raise ImportError(
            "tiktoken is required for accurate token counting. "
            "Install with: pip install tiktoken"
        )

    encoding = tiktoken.encoding_for_model(model_name)

    token_counts = []
    for key, value in data.items():
        prompt = value.get("prompt", "")
        if not isinstance(prompt, str):
            continue

        n_tokens = len(encoding.encode(prompt))
        token_counts.append((key, n_tokens))

    token_counts.sort(key=lambda x: x[1], reverse=True)
    return token_counts[:10]


def build_prompt_dataset(
    model_id_list_path: Union[str, Path],
    output_json_path: Union[str, Path],
    prompt_template: str = "these are the batch samples {sample}",
    sample_joiner: str = "\n",
    token_limit: int = 1000,
    candidate_limit: int = 120,
    model_name: str = "gpt-4o-mini",
    ensure_ascii: bool = False,
    indent: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """
    Reads a JSON file containing a list of noun phrases (strings), chunks them into batches,
    and writes an output dict like:
      {
        "batch_1": {"prompt": "..."},
        "batch_2": {"prompt": "..."},
        ...
      }
    where each prompt is prompt_template.format(sample=<joined batch samples>).

    Returns the built dict as well.
    """

    prompt_template = """
You are filtering noun phrases from AI model descriptions.

Goal: decide whether each phrase states an informative property of an AI model.

Labels:
- keep: clearly states a specific property
- drop: clearly generic, structural, or non-informative

KEEP if the phrase explicitly describes:
- a capability or task
- a modality
- a model class
- a specific architecture, method, or training approach
- deployment, tooling, or interfaces (e.g., CLI, apps)
- safety or alignment mechanisms
- a named benchmark, evaluation, or performance claim
- a concrete license or access condition
- a method, format, or algorithm used in ML

DROP if the phrase:
- only names an object or category (e.g., model, dataset, benchmark, metric, weight)
- is a container, section header, or referential term
- is generic or marketing
- is a business concept, role, or person (not a tool)
- is a vague adjective without specific technical meaning

Rules:
- Naming a thing ≠ describing a property.
- Judge only the phrase itself.

Output one JSON object per line with exactly these fields:
- phrase (exact)
- label ("keep", "drop")
- reason (capability, modality, architecture, training, deployment, safety, benchmark, licensing, generic, marketing, product, unclear)

Use technical reasons only for keep.
Be concise. Do not explain.

Phrases:
{sample}
"""

    model_id_list_path = Path(model_id_list_path)
    output_json_path = Path(output_json_path)

    encoding = tiktoken.encoding_for_model(model_name)

    with model_id_list_path.open("r", encoding="utf-8") as f:
        noun_phrases = json.load(f)

    noun_phrases = [str(x) for x in noun_phrases]

    out: Dict[str, Dict[str, Any]] = {}

    batch_num = 0
    idx = 0
    total = len(noun_phrases)

    while idx < total:
        samples = []

        while idx < total and len(samples) < candidate_limit:
            candidate = noun_phrases[idx]

            tentative_prompt = prompt_template.format(
                sample=sample_joiner.join(samples + [candidate])
            )

            if len(encoding.encode(tentative_prompt)) > token_limit:
                break

            samples.append(candidate)
            idx += 1

        # Safety: ensure progress even if one item is huge
        if not samples:
            samples.append(noun_phrases[idx])
            idx += 1

        out[f"{batch_num}"] = {
            "prompt": prompt_template.format(
                sample=sample_joiner.join(samples)
            )
        }

        batch_num += 1

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=ensure_ascii, indent=indent)

    return out

if __name__ == "__main__":
#     build_prompt_dataset(
#     model_id_list_path="2-NP_EXTRACTION/NP_G5_list.json", # bunu eskilerinde filtrelenmis haliyle degistir
#     output_json_path="10-EVALUATION/llm_nps/input_G4.json", # bunuda G0 yapip runla
#     token_limit = 500,
# )
    # # # Example usage
    LLAMA_API_KEY = ""
    SAVE_LOC = "10-EVALUATION/llm_nps/"
    VERSION = "G4"

    output_path = f"{SAVE_LOC}output_{VERSION}.json"
    checkpoint_path = f"{SAVE_LOC}checkpoints_{VERSION}.json"

    client = LlamaClient(
        api_key=LLAMA_API_KEY,
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )

    with open(f"{SAVE_LOC}input_{VERSION}.json", "r", encoding="utf-8") as f:
        sample_dict = json.load(f)
    lengths = [len(v["prompt"]) for v in sample_dict.values()]
    print(
        f"min={min(lengths)}, "
        f"max={max(lengths)}, "
        f"over_2k={sum(l > 5000 for l in lengths)}"
    )

    top10 = top_10_longest_prompts_by_tokens(sample_dict)
    print(top10)

    start_idx = client.last_checkpoint() + 1

    for idx in range(start_idx, len(sample_dict)):
        item = sample_dict[str(idx)]
        client.prompt(
            item["prompt"],
            key=str(idx),
            meta={k: v for k, v in item.items() if k != "prompt"},
        )
        print(f"Processed {idx}")
