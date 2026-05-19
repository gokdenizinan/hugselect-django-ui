
import json
from E_utils import parse_llm_json_flex  # adjust import
from EA_Features import QualityFeatures
from EB_LLM_Client import LLMClient, LLMResponse, LoggingLLMClient
from EA_Features import (CategoricalFeature, NumericFeature, BoolFeature, RecencyFeature)
from EA_Features import (get_llm_text, to_categorical_feat, to_numeric_feat, to_bool_feat, to_recency_feat)

class QualityFeaturesExtractor:
    def __init__(self, llm_client):
        self.llm = llm_client

    def extract(self, user_text: str) -> QualityFeatures:
        """
        Extract essential features from user input using LLM.
        Expect JSON structured output.
        """
        prompt = f"""
    Extract quality importance ratings (1–5) from the user's text, based on ISO 25000 characteristics.

    USER MESSAGE:
    {user_text}

    Return ONLY valid JSON with the following fields (each: integer 1–5, or null if not known):

    Functional_Suitability
    Compatibility
    Performance_Efficiency
    Reliability
    Interaction_Capability
    Security
    Maintainability
    Flexibility

    SCORING RULES (interpretation):
    - 5 = critical / must-have
    - 4 = important
    - 3 = desirable / normal
    - 2 = low importance
    - 1 = explicitly unimportant
    - null = not mentioned or no clear signal

    INTERPRETATION HINTS (use when text implies a quality):
    - Functional_Suitability: correctness, completeness, accuracy.
    - Compatibility: integrations, cross-platform, co-existence.
    - Performance_Efficiency: speed, throughput, scalability.
    - Reliability: uptime, robustness, fault tolerance.
    - Interaction_Capability: usability, UX, accessibility.
    - Security: confidentiality, privacy, compliance.
    - Maintainability: modifiability, testability, long-term sustainability.
    - Flexibility: adaptability, reusability, configurability.

    OTHER RULES:
    - If explicitly prioritized, assign higher score.
    - If explicitly deprioritized, assign lower score.
    - If not mentioned, use null.

    RESPONSE REQUIREMENTS:
    - Return ONLY JSON.
    - Each field must exist.
    - No comments or explanations.

    Example structure:

    {{
    "Functional_Suitability": <int-or-null>,
    "Compatibility": <int-or-null>,
    "Performance_Efficiency": <int-or-null>,
    "Reliability": <int-or-null>,
    "Interaction_Capability": <int-or-null>,
    "Security": <int-or-null>,
    "Maintainability": <int-or-null>,
    "Flexibility": <int-or-null>
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

        return QualityFeatures(
            Functional_Suitability=data.get("Functional_Suitability"),
            Compatibility=data.get("Compatibility"),
            Performance_Efficiency=data.get("Performance_Efficiency"),
            Reliability=data.get("Reliability"),
            Interaction_Capability=data.get("Interaction_Capability"),
            Security=data.get("Security"),
            Maintainability=data.get("Maintainability"),
            Flexibility=data.get("Flexibility"),
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
        save_dir="8-CRITERIA_SELECTION/user_intent/quality_features",
        print_output=True,
        save_file= "deneme1.json",   # optional
    )

    extractor = QualityFeaturesExtractor(logger)

    user_text = "I want an object detection model that is very fast and can run on mobile devices, but accuracy is not that important. Also, it should be easy to update the model in the future."

    features = extractor.extract(user_text)# RUNLAMAK ICIN UNCOMMENTLE


    #=== ELASTIC SEARCH CLIENT === 
    print("importing creating feature bundle...")

    import json
    import glob

    files = glob.glob("8-CRITERIA_SELECTION/user_intent/quality_features/deneme1.json")

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
        print(features["Functional_Suitability"])

