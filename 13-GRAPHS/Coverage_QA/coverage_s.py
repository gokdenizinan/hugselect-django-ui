from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# BU QUALITY ATTRIBUTE METHODS SECTIONDAKI BUYUK TABLOYU DOLDURMAK ICIN
# h_coun check folders i runla bide sonra karsilastir hangisi iyi
# sayilari tabloyu doldur ona gore
SOURCE_ORDER = ["stackoverflow", "reddit", "hf"]
SOURCE_LABELS = {
    "stackoverflow": "StackOverflow",
    "reddit": "Reddit",
    "hf": "Hugging Face",
}


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_nested_review_json(
    json_path: str | Path,
    drop_topic_xyg: bool = True,
) -> pd.DataFrame:
    """
    Loads nested review JSON like:
        llm_check_reviews/split_1_no_name.json

    Expected structure per model:
    {
        "model_id": "...",
        "topic": "...",
        "reddit": [{"score": ..., "mentioned": [...]}, ...],
        "hf": [...],
        "stackoverflow": [...]
    }

    Returns a flat DataFrame with columns:
        model_id, topic, source, score, reviews
    """
    data = load_json(json_path)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in JSON: {json_path}")

    rows: list[dict[str, Any]] = []

    for model_entry in data:
        model_id = model_entry.get("model_id")
        topic = model_entry.get("topic")

        for source_name in ["reddit", "hf", "stackoverflow"]:
            for entry in model_entry.get(source_name, []) or []:
                score = entry.get("score")
                mentioned_texts = entry.get("mentioned", []) or []

                for text in mentioned_texts:
                    rows.append(
                        {
                            "model_id": model_id,
                            "topic": topic,
                            "source": source_name,
                            "score": score,
                            "reviews": text,
                        }
                    )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["model_id"] = df["model_id"].fillna("").astype(str)
    df["topic"] = df["topic"].fillna("").astype(str)
    df["source"] = df["source"].fillna("").astype(str)
    df["reviews"] = df["reviews"].fillna("").astype(str)

    if drop_topic_xyg and "topic" in df.columns:
        df = df[df["topic"] != "xyg"].copy()

    return df.reset_index(drop=True)


def load_tabular_dataset(
    file_path: str | Path,
    drop_topic_xyg: bool = True,
) -> pd.DataFrame:
    """
    Loads a flat CSV/JSON dataset with columns like:
        model_id, topic, source, reviews, ...

    Supports:
        .csv
        .json (records-style list)
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() == ".json":
        data = load_json(file_path)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    for col in ["model_id", "topic", "source", "reviews"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if drop_topic_xyg and "topic" in df.columns:
        df = df[df["topic"] != "xyg"].copy()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------

def load_top100_model_ids(popular_json_path: str | Path) -> list[str]:
    popular = load_json(popular_json_path)

    if not isinstance(popular, dict):
        raise ValueError(f"Expected dict-like JSON in {popular_json_path}")

    return list(popular.keys())[:100]


def dataset_stats_table(
    df: pd.DataFrame,
    top100_model_ids: list[str],
    source_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Returns a table with rows:
        StackOverflow / Reddit / Hugging Face / Total

    and columns:
        Reviews, Models, M100
    """
    if source_order is None:
        source_order = SOURCE_ORDER

    rows = []

    for source in source_order:
        part = df[df["source"] == source].copy() if "source" in df.columns else pd.DataFrame()

        reviews = len(part)
        models = part["model_id"].nunique() if "model_id" in part.columns else 0
        m100 = part[part["model_id"].isin(top100_model_ids)]["model_id"].nunique() if "model_id" in part.columns else 0

        rows.append(
            {
                "Source": SOURCE_LABELS.get(source, source),
                "Reviews": reviews,
                "Models": models,
                "M100": m100,
            }
        )

    total_reviews = len(df)
    total_models = df["model_id"].nunique() if "model_id" in df.columns else 0
    total_m100 = df[df["model_id"].isin(top100_model_ids)]["model_id"].nunique() if "model_id" in df.columns else 0

    rows.append(
        {
            "Source": "Total",
            "Reviews": total_reviews,
            "Models": total_models,
            "M100": total_m100,
        }
    )

    return pd.DataFrame(rows)


def print_stage_report(
    stage_name: str,
    df: pd.DataFrame,
    top100_model_ids: list[str],
) -> None:
    print("\n" + "=" * 100)
    print(stage_name)
    print("=" * 100)

    print(f"Total rows   : {len(df):,}")
    if "model_id" in df.columns:
        print(f"Unique models: {df['model_id'].nunique():,}")
    if "reviews" in df.columns:
        print(f"Unique reviews: {df['reviews'].nunique():,}")

    table = dataset_stats_table(df, top100_model_ids)
    print()
    print(table.to_string(index=False))


def print_latex_block(stage_name: str, table: pd.DataFrame) -> None:
    """
    Prints rows you can paste into LaTeX manually if needed.
    """
    print("\n" + "-" * 100)
    print(f"LaTeX-like rows for: {stage_name}")
    print("-" * 100)

    for _, row in table.iterrows():
        print(
            f"{row['Source']} & {row['Reviews']} & {row['Models']} & {row['M100']} \\\\"
        )


# ---------------------------------------------------------------------
# Stage-specific wrappers
# ---------------------------------------------------------------------

def report_nested_json_stage(
    stage_name: str,
    json_path: str | Path,
    popular_json_path: str | Path,
    drop_topic_xyg: bool = True,
) -> pd.DataFrame:
    df = load_nested_review_json(json_path, drop_topic_xyg=drop_topic_xyg)
    top100_model_ids = load_top100_model_ids(popular_json_path)
    print_stage_report(stage_name, df, top100_model_ids)
    table = dataset_stats_table(df, top100_model_ids)
    print_latex_block(stage_name, table)
    return table


def report_tabular_stage(
    stage_name: str,
    file_path: str | Path,
    popular_json_path: str | Path,
    drop_topic_xyg: bool = True,
) -> pd.DataFrame:
    df = load_tabular_dataset(file_path, drop_topic_xyg=drop_topic_xyg)
    top100_model_ids = load_top100_model_ids(popular_json_path)
    print_stage_report(stage_name, df, top100_model_ids)
    table = dataset_stats_table(df, top100_model_ids)
    print_latex_block(stage_name, table)
    return table


def combine_stage_tables(stage_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combines multiple stage tables into one wide table:
        Source | F2 Reviews | F2 Models | F2 M100 | F3 Reviews | ...
    """
    merged = None

    for stage_name, df in stage_tables.items():
        renamed = df.rename(
            columns={
                "Reviews": f"{stage_name} Reviews",
                "Models": f"{stage_name} Models",
                "M100": f"{stage_name} M100",
            }
        )

        if merged is None:
            merged = renamed
        else:
            merged = merged.merge(renamed, on="Source", how="outer")

    return merged if merged is not None else pd.DataFrame()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    # Change only this if your thesis folder name/path is different.
    BASE_DIR = Path("/Users/ardacanseradali/Documents/Tez_master/6-REVIEW_SENTIMENT_ANALYSIS")

    # Popular top-100 model list
    POPULAR_JSON = BASE_DIR / "N_model_likes_10k_P9.json"

    # Visible in notebook
    FILTER2_JSON = BASE_DIR / "llm_check_reviews" / "split_1_no_name.json"
    FILTER3_CSV = BASE_DIR / "llm_check_meaning" / "preprocessing_sentiment_go.csv"
    FILTER4_CSV = BASE_DIR / "llm_check_meaning" / "sentiment_for_f5_united_3_llm_go.csv"

    stage_tables: dict[str, pd.DataFrame] = {}

    # Filter 2 - LLM checker
    stage_tables["Filter2"] = report_nested_json_stage(
        stage_name="Filter 2 - LLM checker",
        json_path=FILTER2_JSON,
        popular_json_path=POPULAR_JSON,
        drop_topic_xyg=True,
    )

    # Filter 3 - Transformer checker
    stage_tables["Filter3"] = report_tabular_stage(
        stage_name="Filter 3 - Transformer checker",
        file_path=FILTER3_CSV,
        popular_json_path=POPULAR_JSON,
        drop_topic_xyg=True,
    )

    # Filter 4 - Sentiment checker
    stage_tables["Filter4"] = report_tabular_stage(
        stage_name="Filter 4 - Sentiment checker",
        file_path=FILTER4_CSV,
        popular_json_path=POPULAR_JSON,
        drop_topic_xyg=True,
    )

    summary = combine_stage_tables(stage_tables)

    print("\n" + "#" * 100)
    print("COMBINED STAGE SUMMARY")
    print("#" * 100)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()