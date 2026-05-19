from dataclasses import dataclass
from typing import Any

@dataclass
class LLMResponse:
    text: str
    finish_reason: str
    raw: Any          # full raw response from Gemini (if you need it)


import time
from typing import Optional
from google import genai


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 5,
        retry_delay_seconds: float = 20.0,
    ):
        """
        Thin wrapper around google.genai.Client for chat / completion-style usage.
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def generate(
        self,
        prompt: str,
        require_stop: bool = True,
    ) -> LLMResponse:
        """
        Send a single prompt to the LLM and return the text.

        - Retries on exceptions up to max_retries
        - Optionally enforces finish_reason == "STOP"
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )

                # Basic safety checks
                if not response.candidates:
                    raise RuntimeError("No candidates returned from Gemini")

                candidate = response.candidates[0]
                finish_reason = getattr(candidate, "finish_reason", None)

                # Enforce STOP if requested
                if require_stop and finish_reason != "STOP":
                    raise RuntimeError(
                        f"Unexpected finish_reason: {finish_reason}"
                    )

                text = getattr(response, "text", None)
                if text is None:
                    # Fallback: concatenate parts if needed
                    parts = getattr(candidate, "content", None)
                    if parts and getattr(parts, "parts", None):
                        text = "".join(p.text for p in parts.parts)
                    else:
                        raise RuntimeError("No text in response")

                return LLMResponse(
                    text=str(text),
                    finish_reason=str(finish_reason),
                    raw=response,
                )

            except Exception as e:
                last_error = e
                if attempt >= self.max_retries:
                    # Give up and re-raise the last error
                    raise
                # Wait before retrying
                time.sleep(self.retry_delay_seconds)

        # If we exit the loop without returning or raising, raise last error
        raise RuntimeError(f"LLMClient failed after retries: {last_error}")


import os
import json
import time
from typing import Optional, Dict, Any


class LoggingLLMClient:
    """
    Decorator/wrapper around your existing LLMClient.
    Logs every LLM response to disk for offline re-parsing and debugging.
    """

    def __init__(
        self,
        llm_client,
        save_dir: str = "logs/llm_outputs",
        print_output: bool = True,
        save_file: Optional[str] = None,  # save combined dict into this file
    ):
        self.llm = llm_client
        self.print_output = print_output

        # create directory once
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # combined file path
        self.save_file = (
            os.path.join(save_dir, save_file) if save_file else None
        )

        self.batch_results = []  # store results in-memory if saving as one file

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Calls underlying LLMClient.generate(prompt) and logs the result.

        Returns a dict:
        {
            "prompt": str,
            "response": str,
        }
        """
        # Call underlying client
        response = self.llm.generate(prompt, **kwargs)
        text = response.text

        # Bundle result with useful metadata
        record = {
            "prompt": prompt,
            "response": text,
            "finish_reason": response.finish_reason,
        }

        # ---- PRINT ----
        if self.print_output:
            print("\n=== LLM OUTPUT START ===")
            print(text)
            print("=== LLM OUTPUT END ===\n")

        # ---- SAVE BATCH JSON ----
        if self.save_file:
            # key = str(len(self.batch_results))
            self.batch_results.append(record)
            with open(self.save_file, "w", encoding="utf-8") as f:
                json.dump(self.batch_results, f, indent=2, ensure_ascii=False)

        return record


if __name__ == "__main__":
    import os
    import json
    import time
    from tqdm import tqdm
    from collections import defaultdict
    import copy

    # ----- your constants / paths -----
    SAVE_LOC = "8-CRITERIA_SELECTION/user_intent/"
    VERSION = "D"
    C_NAME = "testing_gemini"

    # Load checkpoint (same as you had)
    save_nums = list(range(20, 0, -1))
    for i in save_nums:
        SAVE_NUM = i
        check_path = f"{SAVE_LOC}checkpoints_{VERSION}_{SAVE_NUM-1}.json"
        if os.path.exists(check_path):
            with open(check_path, "r", encoding="utf-8") as f:
                checkpoints = json.load(f)
            CHECKPOINT = checkpoints[C_NAME]
            break
        else:
            SAVE_NUM = 0
            CHECKPOINT = 0

    print(f"Save num: {SAVE_NUM}/{len(save_nums)}, checkpoint = {CHECKPOINT}")
    print("Loading input dict...")

    sample_dict = [
        {
            "query_type": "testing",
            "prompt": "can you count to 10?"
        }
    ]

    SAVE_FIL = f"quality_mapping_output_{VERSION}{SAVE_NUM}"
    CHECK_LOC = f"checkpoints_{VERSION}{SAVE_NUM}"

    # ----- NEW: create LLMClient -----
    GEMINI_API_KEY = ""

    # from your_module import LLMClient  # adjust import
    # from your_module import LLMResponse

    llm_client = LLMClient(
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.5-flash",
        max_retries=5,
        retry_delay_seconds=20.0,
    )

# ----- main loop -----
if __name__ == "__main__":
    input_dict = sample_dict[CHECKPOINT:]
    results = {}
    hit_error = defaultdict(lambda: defaultdict(int))

    success = 0
    name_keys = [str(i) for i in range(0, len(input_dict) + 2)]

    for key, item in enumerate(tqdm(input_dict, desc="Processing input dict", unit="NP"), start=1):
        prompt = item["prompt"]
        key += CHECKPOINT
        checkpoints = {C_NAME: key}
        name_key = name_keys[key - 1]

        try:
            response = llm_client.generate(prompt=prompt, require_stop=True)

            # On success:
            description = response.text
            results[key] = copy.deepcopy(item)
            results[key]["output"] = str(description)
            results[key]["model_used"] = "GEMINI"
            success += 1

            tqdm.write(f"Success for key {name_key} with GEMINI. Successes in a row: {success}")

            # Save progress
            with open(f"{SAVE_LOC + SAVE_FIL}_U.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            with open(f"{SAVE_LOC + CHECK_LOC}.json", "w", encoding="utf-8") as f:
                json.dump(checkpoints, f, indent=2, ensure_ascii=False)

            if success >= 1:
                time.sleep(1)
            else:
                tqdm.write("sleep 10 seconds")
                time.sleep(10)

        except Exception as e:
            success = 0
            tqdm.write(f"Exception for key {name_key} with GEMINI: {e}")
            hit_error[key]["Gemini_Exception"] += 1
            # (optional) extra delay here if you want

    # Final save
    with open(f"{SAVE_LOC + SAVE_FIL}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(f"{SAVE_LOC + CHECK_LOC}.json", "w", encoding="utf-8") as f:
        json.dump(checkpoints, f, indent=2, ensure_ascii=False)

    with open(f"{SAVE_LOC + SAVE_FIL}_hit_error.json", "w", encoding="utf-8") as f:
        json.dump(hit_error, f, indent=2, ensure_ascii=False)

    print("---------- SUMMARY ----------")
    print("Input sentences processed:", len(sample_dict))
    print("Errors encountered:", len(hit_error))
    print("----------------------------------")



