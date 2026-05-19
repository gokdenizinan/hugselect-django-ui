
import json
from E_utils import parse_llm_json_flex  # adjust import
from EA_Features import EssentialFeatures
from EB_LLM_Client import LLMClient, LLMResponse, LoggingLLMClient
from EA_Features import (CategoricalFeature, NumericFeature, BoolFeature, RecencyFeature)
from EA_Features import (get_llm_text, to_categorical_feat, to_numeric_feat, to_bool_feat, to_recency_feat)

class EssentialFeaturesExtractor:
    def __init__(self, llm_client):
        self.llm = llm_client

    def extract(self, user_text: str) -> EssentialFeatures:
        """
        Extract essential features from user input using LLM.
        Expect JSON structured output.
        """
        prompt = f"""
    You will extract structured information from user requirements about Hugging Face models.

    USER MESSAGE: {user_text}

    You must return a single JSON object with the following top-level fields:

    - task: CategoricalFeature or null
    - domain: CategoricalFeature or null
    - model_name: CategoricalFeature or null
    - author: CategoricalFeature or null
    - objective: CategoricalFeature or null
    - task_aliases: CategoricalFeature or null
    - domain_aliases: CategoricalFeature or null

    DEFINITION of CategoricalFeature:
    - include: list of strings (values the user wants)
    - exclude: list of strings (values the user rejects)
    - priority: one of "must", "strong_prefer", "prefer", "avoid", or null

    INTERPRETATION RULES:
    - Only encode preferences that are explicitly stated or very strongly implied.
    - Do NOT guess or invent preferences that are not supported by the user message.
    - If there is no information for a top-level field, set that field to null (not an empty object).
    - For each CategoricalFeature, "include" and "exclude" MUST always be present as lists (possibly empty).
    - Use "priority" to indicate how important the preference seems for the user:
    - "must": absolutely required or hard constraint
    - "strong_prefer": very important but not strictly required
    - "prefer": mild preference
    - "avoid": user explicitly dislikes or rejects something
    - null: when importance is unclear

    OUTPUT REQUIREMENTS:
    - Return ONLY valid, minifiable JSON.
    - Do NOT include any extra keys, comments, natural language, or Markdown.
    - All six top-level fields (task, domain, model_name, author, objective, task_aliases, domain_aliases) MUST be present.
    - Fields with no information MUST be null at the top level.
    - CategoricalFeature.include and CategoricalFeature.exclude MUST always exist as lists (even if empty).
    - CategoricalFeature.priority MUST be one of the allowed values or null.

    Example output format (structure only, example values are placeholders):

    {{
    "task": {{
        "include": ["text-classification"],
        "exclude": [],
        "priority": "must"
    }},
    "domain": {{
        "include": ["medical"],
        "exclude": ["legal"],
        "priority": "strong_prefer"
    }},
    "model_name": {{
        "include" ["Llama-3.2-1B"],
        "exclude": [],
        "priotity": "strong_prefer"
    }}
    "author": null,
    "objective": {{
        "include": ["speed", "low-latency"],
        "exclude": [],
        "priority": "prefer"
    }},
    "task_aliases": {{
        "include": ["sentiment analysis"],
        "exclude": [],
        "priority": "prefer"
    }},
    "domain_aliases": null
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
        
        return EssentialFeatures(
            task=to_categorical_feat(data.get("task")),
            domain=to_categorical_feat(data.get("domain")),
            model_name=to_categorical_feat(data.get("model_name")),
            author=to_categorical_feat(data.get("author")),
            objective=to_categorical_feat(data.get("objective")),
            task_aliases=to_categorical_feat(data.get("task_aliases")),
            domain_aliases=to_categorical_feat(data.get("domain_aliases")),
        )


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
        save_dir="8-CRITERIA_SELECTION/user_intent/essential_features",
        print_output=True,
        save_file= "deneme.json",   # optional
    )

    extractor = EssentialFeaturesExtractor(logger)

    user_text = "Machine translation foundational models supporting low-resource languages"

    # features = extractor.extract(user_text) #RUNLAMAK ICIN UNCOMMENTLE


    #=== ELASTIC SEARCH CLIENT === 
    print("importing creating feature bundle...")

    import json
    import glob

    files = glob.glob("8-CRITERIA_SELECTION/user_intent/essential_features/deneme.json")

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
        print(features["task"])

