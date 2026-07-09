import os
import sys
from pathlib import Path
from pprint import pprint

from EA_Features import (
    FeatureBundle,
    FunctionalFeatures,
)

from EB_LLM_Client import LLMClient, LoggingLLMClient

from EC_EssentialFeatureExtractor import EssentialFeaturesExtractor
from EC_PreferenceFeatureExtractor import PreferenceFeaturesExtractor
from EC_QualityFeatureExtractor import QualityFeaturesExtractor
from EC_FunctionalFeatureExtractor import NounPhraseExtractor

try:
    from E_utils import object_to_dict
except ImportError:
    object_to_dict = None


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs" / "live_feature_extraction"


def show_object(title, obj):
    print(f"\n--- {title} ---")
    if object_to_dict is not None:
        try:
            pprint(object_to_dict(obj), width=120)
            return
        except Exception:
            pass

    pprint(obj, width=120)


def main():
    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
    else:
        user_text = (
            "I need a text classification model for Turkish sentiment analysis, "
            "preferably based on BERT and usable commercially."
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Run:\n"
            "export GEMINI_API_KEY='your-key-here'\n"
            "Do not paste your real key into ChatGPT."
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    llm_client = LLMClient(
        api_key=api_key,
        model_name="gemini-2.5-flash",
        max_retries=2,
        retry_delay_seconds=3.0,
    )

    Elogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir=str(LOG_DIR),
        print_output=True,
        save_file="essential_test.json",
    )

    Plogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir=str(LOG_DIR),
        print_output=True,
        save_file="preference_test.json",
    )

    Qlogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir=str(LOG_DIR),
        print_output=True,
        save_file="quality_test.json",
    )

    print("\nUSER TEXT:")
    print(user_text)

    print("\n[1] Extracting essential features...")
    Eextractor = EssentialFeaturesExtractor(Elogger)
    Efeatures = Eextractor.extract(user_text)
    show_object("EssentialFeatures", Efeatures)

    print("\n[2] Extracting preference features...")
    Pextractor = PreferenceFeaturesExtractor(Plogger)
    Pfeatures = Pextractor.extract(user_text)
    show_object("PreferenceFeatures", Pfeatures)

    print("\n[3] Extracting quality features...")
    Qextractor = QualityFeaturesExtractor(Qlogger)
    Qfeatures = Qextractor.extract(user_text)
    show_object("QualityFeatures", Qfeatures)

    print("\n[4] Extracting functional features...")
    Ffeatures = FunctionalFeatures()
    Fextractor = NounPhraseExtractor()
    Ffeatures.add_from_query(user_text, Fextractor)
    show_object("FunctionalFeatures", Ffeatures)

    print("\n[5] Creating FeatureBundle...")
    Fbundle = FeatureBundle(
        essential=Efeatures,
        preferences=Pfeatures,
        quality=Qfeatures,
        functional=Ffeatures,
    )
    show_object("FeatureBundle", Fbundle)

    print(f"\nLogs saved under: {LOG_DIR}")
    print("\nDONE: live feature extraction wrapper worked.")


if __name__ == "__main__":
    main()
