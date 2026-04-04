#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import re
import pandas as pd


# ----------------------------
# Pipeline tag → modality map
# ----------------------------

PIPELINE_TO_MODALITY = {
    # -------- TEXT --------
    "text-classification": "text",
    "text-generation": "text",
    "text2text-generation": "text",
    "token-classification": "text",
    "fill-mask": "text",
    "sentence-similarity": "text",
    "multiple-choice": "text",
    "zero-shot-classification": "text",
    "question-answering": "text",
    "summarization": "text",
    "translation": "text",
    "text-ranking": "text",
    "text-retrieval": "text",

    # -------- IMAGE --------
    "image-classification": "image",
    "object-detection": "image",
    "image-segmentation": "image",
    "image-feature-extraction": "image",
    "depth-estimation": "image",
    "keypoint-detection": "image",
    "zero-shot-image-classification": "image",
    "zero-shot-object-detection": "image",
    "mask-generation": "image",
    "unconditional-image-generation": "image",
    "image-to-image": "image",

    # -------- AUDIO --------
    "audio-classification": "audio",
    "automatic-speech-recognition": "audio",
    "text-to-speech": "audio",
    "audio-to-audio": "audio",
    "voice-activity-detection": "audio",

    # -------- VIDEO --------
    "video-classification": "video",

    # -------- TABULAR --------
    "tabular-classification": "tabular",
    "tabular-regression": "tabular",
    "table-question-answering": "tabular",
    "table-to-text": "tabular",

    # -------- TIME SERIES --------
    "time-series-forecasting": "time-series",

    # -------- GRAPH --------
    "graph-ml": "graph",

    # -------- MULTIMODAL --------
    "text-to-image": "multimodal",
    "image-to-text": "multimodal",
    "image-text-to-text": "multimodal",
    "visual-question-answering": "multimodal",
    "document-question-answering": "multimodal",
    "visual-document-retrieval": "multimodal",
    "text-to-video": "multimodal",
    "image-to-video": "multimodal",
    "video-to-video": "multimodal",
    "video-text-to-text": "multimodal",
    "text-to-audio": "multimodal",
    "audio-text-to-text": "multimodal",
    "image-to-3d": "multimodal",
    "text-to-3d": "multimodal",
    "any-to-any": "multimodal",

    # -------- EDGE CASES --------
    "feature-extraction": "image",
    "reinforcement-learning": "reinforcement learning",
    "robotics": "robotics",
    "other": "Other / Unclear",
}


# ----------------------------
# Task aliases from tags
# ----------------------------

TASK_GROUPS = {
    # -------- TEXT GENERATION --------
    "text-generation": [
        "text-generation",
        "text2text-generation"
    ],

    "fill-mask": [
        "fill-mask"
    ],

    # -------- TEXT UNDERSTANDING --------
    "text-classification": [
        "text-classification",
        "zero-shot-classification"
    ],

    "token-classification": [
        "token-classification"
    ],

    "sentence-similarity": [
        "sentence-similarity"
    ],

    "multiple-choice": [
        "multiple-choice"
    ],

    # -------- QUESTION ANSWERING --------
    "question-answering": [
        "question-answering"
    ],

    "document-question-answering": [
        "document-question-answering"
    ],

    "table-question-answering": [
        "table-question-answering"
    ],

    "visual-question-answering": [
        "visual-question-answering"
    ],

    # -------- RETRIEVAL / RANKING --------
    "text-retrieval": [
        "text-retrieval"
    ],

    "text-ranking": [
        "text-ranking"
    ],

    "visual-document-retrieval": [
        "visual-document-retrieval"
    ],

    # -------- TEXT TRANSFORM --------
    "translation": [
        "translation"
    ],

    "summarization": [
        "summarization"
    ],

    # -------- IMAGE GENERATION --------
    "text-to-image": [
        "text-to-image"
    ],

    "image-to-image": [
        "image-to-image"
    ],

    "unconditional-image-generation": [
        "unconditional-image-generation"
    ],

    # -------- IMAGE UNDERSTANDING --------
    "image-classification": [
        "image-classification",
        "zero-shot-image-classification"
    ],

    "object-detection": [
        "object-detection",
        "zero-shot-object-detection"
    ],

    "image-segmentation": [
        "image-segmentation"
    ],

    "mask-generation": [  # ✅ added (distinct task)
        "mask-generation"
    ],

    "depth-estimation": [
        "depth-estimation"
    ],

    "keypoint-detection": [
        "keypoint-detection"
    ],

    "image-feature-extraction": [  # ✅ added
        "image-feature-extraction",
    ],

    "feature-extraction": [ 
        "feature-extraction"
    ],
    # -------- IMAGE-TEXT --------
    "image-to-text": [
        "image-to-text"
    ],

    "image-text-to-text": [
        "image-text-to-text"
    ],

    # -------- AUDIO --------
    "automatic-speech-recognition": [
        "automatic-speech-recognition"
    ],

    "text-to-speech": [
        "text-to-speech"
    ],

    "audio-classification": [
        "audio-classification"
    ],

    "voice-activity-detection": [
        "voice-activity-detection"
    ],

    "audio-to-audio": [  # ✅ added
        "audio-to-audio"
    ],

    # -------- MULTIMODAL GENERATION --------
    "text-to-video": [
        "text-to-video"
    ],

    "image-to-video": [
        "image-to-video"
    ],

    "video-to-video": [
        "video-to-video"
    ],

    "text-to-3d": [
        "text-to-3d"
    ],

    "image-to-3d": [
        "image-to-3d"
    ],

    # -------- VIDEO --------
    "video-classification": [
        "video-classification"
    ],

    "video-text-to-text": [
        "video-text-to-text"
    ],

    # -------- TABULAR --------
    "tabular-classification": [
        "tabular-classification"
    ],

    "tabular-regression": [
        "tabular-regression"
    ],

    "table-to-text": [  # ✅ added
        "table-to-text"
    ],

    # -------- TIME SERIES --------
    "time-series-forecasting": [
        "time-series-forecasting"
    ],

    "text-to-audio": [  # ✅ added
        "text-to-audio"
    ],

    "audio-text-to-text": [  # ✅ added
        "audio-text-to-text"
    ],

    "reinforcement-learning": [
        "reinforcement-learning"
    ],

    "robotics": [
        "robotics"
    ],

    "graph-ml": [
        "graph-ml"
    ],

    "any-to-any": [
        "any-to-any"
    ],

    "other": [
        "other"
    ]
}

# ----------------------------
# Build reverse mapping
# ----------------------------

PIPELINE_TO_TASK = {}

for task_group, pipeline_list in TASK_GROUPS.items():
    for p in pipeline_list:
        PIPELINE_TO_TASK[p] = task_group

SPECIFICITY_EXCLUSIONS = {"any-to-any", "other"}

def assign_task(effective_pipeline_tag: str) -> str:
    if pd.isna(effective_pipeline_tag):
        return "other"

    tag = str(effective_pipeline_tag).strip().lower()
    return PIPELINE_TO_TASK.get(tag, "other")

def normalize_text(value) -> str:
    return str(value).strip().lower()


def parse_tags(tags_value):
    if pd.isna(tags_value):
        return []

    if isinstance(tags_value, list):
        return [str(x).strip() for x in tags_value if str(x).strip()]

    if isinstance(tags_value, str):
        text = tags_value.strip()
        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except (ValueError, SyntaxError):
                pass

        parts = re.split(r"\s*[|;,]\s*", text)
        return [p.strip() for p in parts if p.strip()]

    return [str(tags_value).strip()]


def infer_specific_pipeline_tag_from_tags(tags_value):
    tags_list = parse_tags(tags_value)

    # First pass: prefer specific tags
    for item in tags_list:
        norm_item = normalize_text(item)
        if norm_item in PIPELINE_TO_MODALITY and norm_item not in SPECIFICITY_EXCLUSIONS:
            return norm_item

    # Second pass: allow generic fallback
    for item in tags_list:
        norm_item = normalize_text(item)
        if norm_item in PIPELINE_TO_MODALITY:
            return norm_item

    return None

def get_effective_pipeline_tag(row) -> str:
    pipeline_tag = row.get("pipeline_tag", None)
    norm_tag = ""

    if not pd.isna(pipeline_tag):
        norm_tag = normalize_text(pipeline_tag)

    # Case 1: missing pipeline_tag → infer from tags
    if not norm_tag:
        inferred = infer_specific_pipeline_tag_from_tags(row.get("tags", None))
        return inferred if inferred else "other"

    # Case 2: any-to-any → try to replace with something more specific from tags
    if norm_tag == "any-to-any":
        inferred = infer_specific_pipeline_tag_from_tags(row.get("tags", None))
        return inferred if inferred else norm_tag

    # Case 3: normal specific pipeline_tag → keep it
    return norm_tag


def assign_modality(effective_pipeline_tag: str) -> str:
    if pd.isna(effective_pipeline_tag):
        return "Other / Unclear"

    tag = normalize_text(effective_pipeline_tag)
    return PIPELINE_TO_MODALITY.get(tag, "Other / Unclear")


def assign_modalities_csv(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    print(df.columns)

    for col in ["model_id", "model_name", "pipeline_tag", "tags"]:
        if col not in df.columns:
            df[col] = ""

    df["effective_pipeline_tag"] = df.apply(get_effective_pipeline_tag, axis=1)
    df["task"] = df["effective_pipeline_tag"].apply(assign_task)
    df["assigned_modality"] = df["effective_pipeline_tag"].apply(assign_modality)

    output_df = df[
        ["model_id", "model_name", "assigned_modality", "task","pipeline_tag", "effective_pipeline_tag", "tags",
    'model_type', 'base_models', 'library_name',  'description', 'short_description']
    ]

    output_df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

    # Optional diagnostics
    any_to_any_rows = (df["pipeline_tag"].astype(str).str.strip().str.lower() == "any-to-any")
    replaced_rows = any_to_any_rows & (df["effective_pipeline_tag"] != "any-to-any")

    print("\nDiagnostics:")
    print("Rows with original pipeline_tag == any-to-any:", int(any_to_any_rows.sum()))
    print("Rows where any-to-any was replaced from tags:", int(replaced_rows.sum()))

    if replaced_rows.any():
        print("\nExamples of replacements:")
        print(df.loc[replaced_rows, ["pipeline_tag", "effective_pipeline_tag", "tags"]].head(10).to_string(index=False))


if __name__ == "__main__":
    input_csv = "7-CLUSTERING_MODELS/hf_models.csv"
    output_csv = "7-CLUSTERING_MODELS/level1_domain_assignments_improved.csv"

    assign_modalities_csv(input_csv, output_csv)