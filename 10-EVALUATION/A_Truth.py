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
import os
from typing import Dict, List


def build_prompt_dataset(
    model_id_list_path: str,
    dict_folder_path: str,
    output_json_path: str,
    *,
    model_id_key: str = "modelID",
    description_key: str = "description",
):
    """
    Builds a prompt dataset by matching model_ids to dictionaries in a folder.

    Args:
        model_id_list_path: Path to JSON file containing a list of model_ids
        dict_folder_path: Folder containing JSON dictionaries
        output_json_path: Path to save the final JSON
        model_id_key: Key name for model ID in dictionary files
        description_key: Key name for description in dictionary files
    """

    # ---------- Load model IDs ----------
    with open(model_id_list_path, "r", encoding="utf-8") as f:
        model_ids: List[str] = json.load(f)

    if not isinstance(model_ids, list):
        raise ValueError("model_id_list JSON must be a list")

    # ---------- Load all dictionaries into memory ----------
    dict_pool = []

    for filename in os.listdir(dict_folder_path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(dict_folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            dict_pool.append(data)

    # ---------- Build output ----------
    output: Dict[str, Dict] = {}
    idx = 0

    for model_id in model_ids:
        for entry in dict_pool:
            if entry.get(model_id_key) == model_id:
                description = entry.get(description_key)

                if not description:
                    break  # found model, but no description
                prompt = (f"""
                    You are an information extraction system.

                    Task:
                    1) Extract ONLY noun phrases (NPs) that are explicit in the text.
                    2) Keep ONLY noun phrases that describe:
                    - functional capabilities (what it can do),
                    - characteristics (properties/traits/specs),
                    - intended use (purpose, target use cases, domains).
                    3) Do NOT add, infer, paraphrase, generalize, or normalize beyond what is written.
                    4) If a candidate feature is not explicitly stated, DROP it.
                    5) Prefer the shortest noun phrase that still preserves the meaning.
                    6) Exclude: verbs/verb phrases, full sentences, marketing fluff, comparisons, company names (unless they are part of a feature), and vague adjectives without a noun head.
                    7) De-duplicate features (case-insensitive), preserve original wording, and keep order of first appearance.

                    Output:
                    Return ONLY valid JSON in the following shape:
                    {{"features": ["<noun phrase>", "..."]}}

                    Text:
                    {description}
                    """)
                output[str(idx)] = {
                    "model_id": model_id,
                    "description": description,
                    "prompt": prompt,
                }
                idx += 1
                break  # move to next model_id

    # ---------- Save ----------
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(output)} entries to {output_json_path}")


if __name__ == "__main__":
    build_prompt_dataset(
    model_id_list_path="10-EVALUATION/llm_ffs/sampled_modelID_glob.json",
    dict_folder_path="HF-Models-T6",
    output_json_path="10-EVALUATION/llm_ffs/input_A5.json",
)
    # # Example usage
    LLAMA_API_KEY = ""
    SAVE_LOC = "10-EVALUATION/llm_ffs/"
    VERSION = "A5"

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
    # lengths = [len(v["prompt"]) for v in sample_dict.values()]
    # print(
    #     f"min={min(lengths)}, "
    #     f"max={max(lengths)}, "
    #     f"over_2k={sum(l > 5000 for l in lengths)}"
    # )


    start_idx = client.last_checkpoint() + 1

    for idx in range(start_idx, len(sample_dict)):
        item = sample_dict[str(idx)]
        client.prompt(
            item["prompt"],
            key=str(idx),
            meta={k: v for k, v in item.items() if k != "prompt"},
        )
        print(f"Processed {idx}")
