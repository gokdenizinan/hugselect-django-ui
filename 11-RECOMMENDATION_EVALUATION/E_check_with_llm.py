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

from typing import Any, Dict, List, Optional
import re

def build_llm_prompt_records(
    scan_results: List[Dict[str, Any]],
    *,
    max_hits_per_paper: int = 5,
    max_chars_per_hit: int = 2000,
    include_hits_when_no_matches: bool = False,
    task_name: str = "Extract paper metadata & rationale",
) -> Dict[int, Dict[str, Any]]:
    """
    Convert scanner output (list of dicts) into {0: {prompt: ...}, 1: {prompt: ...}, ...}
    for LLM processing.

    Expected scan_results item shape (flexible):
      - paper_id or openalex_id
      - title
      - year (optional)
      - doi (optional)
      - pdf_url (optional)
      - status (optional)
      - hits: list of {pages/page_numbers, snippet, matched_keywords (optional)}

    Returns:
      dict indexed from 0..N-1; each value includes:
        - prompt: instruction + relevant context snippets
        - (optional) paper metadata fields copied over
    """

    def _truncate(s: str, n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[: n].rstrip() + "…"

    def _clean_snippet(s: str) -> str:
        # Light cleanup: collapse excessive whitespace but keep paragraph breaks
        s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def _fmt_pages(hit: Dict[str, Any]) -> str:
        pages = hit.get("pages") or hit.get("page_numbers") or []
        if isinstance(pages, (list, tuple)) and pages:
            return f"pp. {', '.join(map(str, pages))}"
        return "page unknown"

    def _fmt_hit(hit: Dict[str, Any], idx: int) -> str:
        snippet = _truncate(_clean_snippet(hit.get("snippet", "")), max_chars_per_hit)
        kw = hit.get("matched_keywords")
        kw_str = f"Matched keywords: {', '.join(kw)}\n" if isinstance(kw, list) and kw else ""
        return (
            f"[EVIDENCE {idx}] ({_fmt_pages(hit)})\n"
            f"{kw_str}"
            f"{snippet}\n"
        )

    out: Dict[int, Dict[str, Any]] = {}
    k = 0

    for r in scan_results:
        paper_id = r.get("paper_id") or r.get("openalex_id") or r.get("id")
        title = r.get("title", "")
        year = r.get("year")
        doi = r.get("doi")
        pdf_url = r.get("pdf_url") or r.get("landing_page_url")
        status = r.get("status", "")

        hits = r.get("hits") or []
        if not hits and not include_hits_when_no_matches:
            # Still generate a prompt, but warn that evidence is empty.
            evidence_block = "No keyword hits/snippets were extracted for this paper.\n"
        else:
            # choose top hits (you can swap to score/order differently if you stored that)
            selected = hits[:max_hits_per_paper]
            if not selected and include_hits_when_no_matches:
                evidence_block = "No hits available.\n"
            else:
                evidence_block = "\n".join(_fmt_hit(h, i + 1) for i, h in enumerate(selected))

        # Prompt: directions + schema + evidence
        prompt = f"""You are an expert research assistant. Task: {task_name}.

Given the paper metadata and evidence excerpts below, extract structured information.
Only use the evidence. If something is not supported by the evidence, output null for it.

Return STRICT JSON with this schema:
{{
  "paper": {{
    "paper_id": string|null,
    "title": string|null,
    "year": number|null,
    "doi": string|null,
    "pdf_url": string|null
  }},
  "domain": {{
    "primary": string|null,
    "secondary": [string]  // can be empty
  }},
  "entities": {{
    "models": [string],
    "datasets": [string],
    "licenses": [string],
    "organizations": [string]
  }},
  "claims": {{
    "training_data": string|null,
    "licensing": string|null,
    "bias_or_fairness": string|null,
    "evaluation_or_benchmarks": string|null
  }},
  "rationale": {{
    "summary": string|null,
    "evidence": [
      {{
        "evidence_id": string,
        "quote": string,
        "why_relevant": string
      }}
    ]
  }}
}}

Rules:
- Keep quotes short (1–2 sentences) and copy them verbatim from evidence snippets.
- "domain.primary" should be a short label like "LLM survey", "dataset documentation", "bias/fairness", "model licensing", etc.
- Put anything uncertain as null.
- If multiple interpretations exist, prefer the one with direct textual support.

PAPER METADATA:
- paper_id: {paper_id}
- title: {title}
- year: {year if year is not None else None}
- doi: {doi if doi else None}
- pdf_url: {pdf_url if pdf_url else None}
- scan_status: {status if status else None}

EVIDENCE EXCERPTS:
{evidence_block}
"""

        out[k] = {
            "prompt": prompt,
            # Helpful extra fields for your pipeline (optional)
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "doi": doi,
            "pdf_url": pdf_url,
            "status": status,
            "hit_count": r.get("hit_count", len(hits)),
        }
        k += 1

    return out




if __name__ == "__main__":
    with open("11-RECOMMENDATION_EVALUATION/paper_model_2/paper_keyword_hits.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    prompt_records = build_llm_prompt_records(results, max_hits_per_paper=400)

    # # # # Example usage
    # LLAMA_API_KEY = ""
    # SAVE_LOC = "10-EVALUATION/llm_nps/"
    # VERSION = "A7"

    # output_path = f"{SAVE_LOC}output_{VERSION}.json"
    # checkpoint_path = f"{SAVE_LOC}checkpoints_{VERSION}.json"

    # client = LlamaClient(
    #     api_key=LLAMA_API_KEY,
    #     model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    #     output_path=output_path,
    #     checkpoint_path=checkpoint_path,
    # )

    # with open(f"{SAVE_LOC}input_{VERSION}.json", "r", encoding="utf-8") as f:
    #     sample_dict = json.load(f)
    # lengths = [len(v["prompt"]) for v in sample_dict.values()]
    # print(
    #     f"min={min(lengths)}, "
    #     f"max={max(lengths)}, "
    #     f"over_2k={sum(l > 5000 for l in lengths)}"
    # )

    # top10 = top_10_longest_prompts_by_tokens(sample_dict)
    # print(top10)

    # start_idx = client.last_checkpoint() + 1

    # for idx in range(start_idx, len(sample_dict)):
    #     item = sample_dict[str(idx)]
    #     client.prompt(
    #         item["prompt"],
    #         key=str(idx),
    #         meta={k: v for k, v in item.items() if k != "prompt"},
    #     )
    #     print(f"Processed {idx}")
