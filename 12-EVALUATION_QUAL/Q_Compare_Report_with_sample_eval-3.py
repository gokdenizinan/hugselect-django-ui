# Q_Compare_Report.py
# Run this file directly in VS Code (Run ▶️). Edit the CONFIG section at the bottom.

from __future__ import annotations

import ast
import csv
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# =========================
# CSV loading (FIXED)
# =========================

def read_table_robust(path: str, preferred_sep: str = "\t") -> pd.DataFrame:
    """
    Robust loader for "CSV" files that are actually TSV or semicolon-separated.

    Your columns look tab-separated:
      ModelID\tReview_Processed\tchatgpt_primary\t...

    This function:
      1) tries preferred_sep (default: tab)
      2) tries auto-detect (python engine)
      3) tries common separators
      4) last resort: QUOTE_NONE + skip bad lines (keeps most rows if lists contain commas)
    """
    errors = []

    # 1) Preferred separator (TAB for your case)
    try:
        df = pd.read_csv(path, sep=preferred_sep)
        # sanity: if we got at least 2 cols, it's likely correct
        if df.shape[1] >= 2:
            return df
    except Exception as e:
        errors.append((f"sep='{preferred_sep}'", e))

    # 2) Auto-detect separator (python engine)
    try:
        df = pd.read_csv(path, engine="python", sep=None)
        if df.shape[1] >= 2:
            return df
    except Exception as e:
        errors.append(("engine='python', sep=None", e))

    # 3) Try common separators
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            errors.append((f"engine='python', sep='{sep}'", e))

    # 4) Last resort: ignore quoting problems and skip bad lines
    for sep in [preferred_sep, ",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                quoting=csv.QUOTE_NONE,
                escapechar="\\",
                on_bad_lines="skip",
            )
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            errors.append((f"QUOTE_NONE + skip bad lines, sep='{sep}'", e))

    msg = "Failed to parse file. Tried:\n" + "\n".join(
        [f"- {name}: {type(err).__name__}: {err}" for name, err in errors]
    )
    raise ValueError(msg)


# =========================
# Evaluation helpers
# =========================

TRUTH_COLS_REQUIRED = [
    "chatgpt_primary",
    "gemini_primary",
    "chatgpt_secondary",
    "gemini_secondary",
]


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def _parse_secondary_cell(x) -> List[str]:
    """
    Secondary columns are lists. Handles:
      - list objects
      - stringified lists: "['a','b']"
      - JSON-ish lists: '["a","b"]'
      - comma/semicolon strings: "a,b" or "a; b"
      - empty/NaN
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []

    # Try parsing list-like strings
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(i).strip() for i in v if str(i).strip()]
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
    except Exception:
        pass

    # Fallback: split
    parts = []
    for token in s.replace(";", ",").split(","):
        t = token.strip()
        if t:
            parts.append(t)
    return parts


@dataclass
class EvalResult:
    evaluation: str
    n_rows_total: int
    n_rows_used: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float


def _compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)

    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision_macro": float(p_mac),
        "recall_macro": float(r_mac),
        "f1_macro": float(f1_mac),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
    }


def _save_confusion_matrix_png(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    outpath: str,
    title: str,
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val == 0:
                continue
            ax.text(
                j, i, str(val),
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=8
            )

    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def _truth_candidates(row: pd.Series) -> Dict[str, List[str]]:
    cprim = _safe_str(row["chatgpt_primary"])
    gprim = _safe_str(row["gemini_primary"])
    csec = _parse_secondary_cell(row["chatgpt_secondary"])
    gsec = _parse_secondary_cell(row["gemini_secondary"])

    return {
        "chatgpt": ([cprim] if cprim else []) + csec,
        "gemini": ([gprim] if gprim else []) + gsec,
    }


def _pick_single_truth_for_cm(
    truth_candidates: List[str],
    pred: str,
    use_secondary_as_truth: bool,
) -> Optional[str]:
    """
    Confusion matrix needs exactly one true label.
    - Primary is truth_candidates[0] by construction.
    - If use_secondary_as_truth and pred matches any candidate -> truth = pred
      (so it counts as correct without forcing an arbitrary primary in CM).
    """
    if not truth_candidates:
        return None

    primary = truth_candidates[0]
    if use_secondary_as_truth and pred and (pred in set(truth_candidates)):
        return pred

    return primary if primary else None


def _evaluate_one(
    df: pd.DataFrame,
    pred_col: str,
    truth_source: str,  # "chatgpt" | "gemini" | "cumulative_agreement_only"
    use_secondary_as_truth: bool,
    output_dir: str,
    eval_name: str,
) -> Tuple[EvalResult, pd.DataFrame]:
    y_true: List[str] = []
    y_pred: List[str] = []

    total = len(df)
    used = 0

    for _, row in df.iterrows():
        pred = _safe_str(row[pred_col])
        if pred == "":
            continue

        # --- NEW: choose candidates based on truth_source ---
        if truth_source == "cumulative_agreement_only":
            cprim = _safe_str(row["chatgpt_primary"])
            gprim = _safe_str(row["gemini_primary"])
            if not (cprim and gprim and cprim == gprim):
                continue

            candidates = [cprim]
            if use_secondary_as_truth:
                candidates = (
                    [cprim]
                    + _parse_secondary_cell(row["chatgpt_secondary"])
                    + _parse_secondary_cell(row["gemini_secondary"])
                )

            truth = _pick_single_truth_for_cm(candidates, pred, use_secondary_as_truth)

        elif truth_source == "union_primary":
            # Union-of-annotators truth (Option 1)
            cprim = _safe_str(row["chatgpt_primary"])
            gprim = _safe_str(row["gemini_primary"])
            if not cprim or not gprim:
                continue

            # If pred matches either, treat as correct by setting truth=pred
            if pred == cprim or pred == gprim:
                truth = pred
            else:
                # If pred matches neither, pick a consistent single truth for CM/report
                # (you can flip this to gprim if you prefer)
                truth = cprim

        else:
            cand_map = _truth_candidates(row)  # chatgpt / gemini, includes secondaries
            candidates = cand_map.get(truth_source, [])
            if not candidates:
                continue
            if not use_secondary_as_truth:
                candidates = candidates[:1]  # primary only

            truth = _pick_single_truth_for_cm(candidates, pred, use_secondary_as_truth)

        if truth is None:
            continue

        y_true.append(truth)
        y_pred.append(pred)
        used += 1

    if used == 0:
        res = EvalResult(
            evaluation=eval_name,
            n_rows_total=total,
            n_rows_used=0,
            accuracy=float("nan"),
            precision_macro=float("nan"),
            recall_macro=float("nan"),
            f1_macro=float("nan"),
            precision_weighted=float("nan"),
            recall_weighted=float("nan"),
            f1_weighted=float("nan"),
        )
        return res, pd.DataFrame()

    metrics = _compute_metrics(y_true, y_pred)
    labels = sorted(list(set(y_true) | set(y_pred)))

    cm_path = os.path.join(output_dir, f"confusion_matrix__{eval_name}.png")
    _save_confusion_matrix_png(
        y_true, y_pred, labels, cm_path,
        title=f"Confusion Matrix — {eval_name} (n={used})"
    )

    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_label_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "label"})
    per_label_df.insert(0, "evaluation", eval_name)

    res = EvalResult(
        evaluation=eval_name,
        n_rows_total=total,
        n_rows_used=used,
        **metrics,
    )
    return res, per_label_df


def evaluate_quality_predictions(
    input_path: str,
    output_dir: str,
    *,
    prediction_col: str = "predicted",
    preferred_sep: str = "\t",
    use_secondary_as_truth: bool = False,
    save_cumulative_truth_set: bool = True,
) -> Dict[str, str]:
    """
    Main entry point (VS Code friendly): edit parameters in code, press Run.

    input_path: your tab/csv file path
    output_dir: where reports + pngs will be written
    prediction_col: default "predicted" (matches your columns)
    preferred_sep: default "\\t" (matches your file)
    use_secondary_as_truth: if True, prediction matching any secondary counts as correct
    """
    os.makedirs(output_dir, exist_ok=True)

    df = read_table_robust(input_path, preferred_sep=preferred_sep)

    # Sanity print (optional; keep if helpful)
    print("Loaded shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # -----------------------------
    # FILTER ROWS (important)
    # keep only rows where chat, gemini and predicted exist
    # -----------------------------
    df = df[
        df["chatgpt_primary"].notna()
        & df["gemini_primary"].notna()
        & df[prediction_col].notna()
    ]

    # remove empty strings
    df = df[
        (df["chatgpt_primary"].astype(str).str.strip() != "")
        & (df["gemini_primary"].astype(str).str.strip() != "")
        & (df[prediction_col].astype(str).str.strip() != "")
    ]

    print("Rows used after filtering:", len(df))

    missing = [c for c in TRUTH_COLS_REQUIRED + [prediction_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out_paths: Dict[str, str] = {}

    # Cumulative truth set: rows where primaries match
    agree_mask = (
        df["chatgpt_primary"].astype(str).str.strip()
        == df["gemini_primary"].astype(str).str.strip()
    )

    if save_cumulative_truth_set:
        cumulative_truth = df.loc[agree_mask, ["chatgpt_primary"]].copy()
        cumulative_truth = cumulative_truth.rename(columns={"chatgpt_primary": "agreed_primary"})
        cum_path = os.path.join(output_dir, "cumulative_truth_set.csv")
        cumulative_truth.to_csv(cum_path, index=False)
        out_paths["cumulative_truth_set_csv"] = cum_path

    evals = [
        ("truth_chatgpt_primary", "chatgpt"),
        ("truth_gemini_primary", "gemini"),
        ("truth_union_primary", "union_primary"),              # ✅ NEW (Option 1)
        # ("truth_cumulative_agreement", "cumulative_agreement_only"),
    ]

    overall_rows = []
    per_label_parts = []

    for eval_name, truth_source in evals:
        suffix = "with_secondary" if use_secondary_as_truth else "primary_only"
        full_name = f"{eval_name}__{suffix}"

        res, per_label_df = _evaluate_one(
            df=df,
            pred_col=prediction_col,
            truth_source=truth_source,
            use_secondary_as_truth=use_secondary_as_truth,
            output_dir=output_dir,
            eval_name=full_name,
        )

        overall_rows.append(res.__dict__)
        if not per_label_df.empty:
            per_label_parts.append(per_label_df)

    overall_df = pd.DataFrame(overall_rows)
    overall_path = os.path.join(output_dir, "overall_evaluation.csv")
    overall_df.to_csv(overall_path, index=False)
    out_paths["overall_evaluation_csv"] = overall_path

    per_label_df = pd.concat(per_label_parts, ignore_index=True) if per_label_parts else pd.DataFrame()
    per_label_path = os.path.join(output_dir, "per_label_evaluation.csv")
    per_label_df.to_csv(per_label_path, index=False)
    out_paths["per_label_evaluation_csv"] = per_label_path

    print("Wrote outputs:")
    for k, v in out_paths.items():
        print(f" - {k}: {v}")
    print(f" - confusion_matrix__*.png in: {output_dir}")

    return out_paths


# =========================
# Sample-level evaluation export
# =========================

