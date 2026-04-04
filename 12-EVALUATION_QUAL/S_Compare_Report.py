import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def _normalize_text(x) -> str:
    """Lowercase, strip, keep letters only (handles 'unclear.', 'Neutral!', etc.)."""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z]+", "", s)
    return s


def normalize_chat_label(x) -> str:
    return _normalize_text(x)


def normalize_pred_label(x) -> str:
    s = _normalize_text(x)
    return "neutral" if s == "unclear" else s


def build_cumulative(chatgpt_label: str, gemini_label: str) -> str:
    # If chatgpt and gemini disagree -> neutral
    if chatgpt_label and (chatgpt_label == gemini_label):
        return chatgpt_label
    return "neutral"


def _read_table_auto(path: str) -> pd.DataFrame:
    """
    Robust delimiter detection for comma/tab/semicolon-separated files.
    """
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        pass

    for sep in [",", "\t", ";", "|"]:
        try:
            return pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            continue

    raise ValueError(
        "Could not parse the file. It may have inconsistent delimiters or unescaped quotes."
    )


def _drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where chatgpt, gemini, predicted are all present and non-empty.
    """
    df2 = df.dropna(subset=["chatgpt", "gemini", "predicted"]).copy()

    # Also drop rows where any required field is empty/whitespace
    for col in ["chatgpt", "gemini", "predicted"]:
        df2[col] = df2[col].astype(str)
    mask = (
        df2["chatgpt"].str.strip().ne("") &
        df2["gemini"].str.strip().ne("") &
        df2["predicted"].str.strip().ne("")
    )
    return df2.loc[mask].copy()


def _save_confusion_matrix_png(cm, labels, output_png_path: str):
    """
    Saves a confusion matrix plot as PNG.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), max(5, len(labels) * 1.0)))
    im = ax.imshow(cm)  # default colormap (no manual color specification)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Annotate counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_png_path, dpi=200)
    plt.close(fig)


def evaluate_sentiment_csv(
    input_path: str,
    overall_csv_path: str,
    per_label_csv_path: str,
    confusion_png_path: str,
    output_evaluated_rows_path: str | None = None,
):
    """
    Input CSV columns: chatgpt, gemini, predicted

    Rules:
    - cumulative truth = chatgpt if chatgpt==gemini else neutral
    - predicted: "unclear" -> neutral
    - only evaluate rows where all three columns have values

    Outputs:
    - overall metrics CSV (single-row)
    - per-label metrics CSV
    - confusion matrix PNG
    - optional evaluated rows CSV (for auditing)
    """

    df = _read_table_auto(input_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"chatgpt", "gemini", "predicted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}. Found: {list(df.columns)}")

    df = _drop_incomplete_rows(df)
    if df.empty:
        raise ValueError("After filtering incomplete rows, no rows remained to evaluate.")

    # Normalize labels
    df["chatgpt_n"] = df["chatgpt"].map(normalize_chat_label)
    df["gemini_n"] = df["gemini"].map(normalize_chat_label)
    df["pred_n"] = df["predicted"].map(normalize_pred_label)

    # Build cumulative truth
    df["cumulative"] = [
        build_cumulative(cg, gm) for cg, gm in zip(df["chatgpt_n"], df["gemini_n"])
    ]

    y_true = df["cumulative"].tolist()
    y_pred = df["pred_n"].tolist()

    labels = sorted((set(y_true) | set(y_pred) | {"neutral"}) - {""})

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    cr = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # --- Overall CSV (metrics at top; single row) ---
    overall_row = {
        "n_rows_used": int(len(df)),
        "accuracy": acc,
        "precision_macro": cr["macro avg"]["precision"],
        "recall_macro": cr["macro avg"]["recall"],
        "f1_macro": cr["macro avg"]["f1-score"],
        "precision_weighted": cr["weighted avg"]["precision"],
        "recall_weighted": cr["weighted avg"]["recall"],
        "f1_weighted": cr["weighted avg"]["f1-score"],
        "support_total": cr["macro avg"]["support"],
    }
    overall_df = pd.DataFrame([overall_row])
    overall_df.to_csv(overall_csv_path, index=False)

    # --- Per-label CSV (one row per label) ---
    per_label_rows = []
    for lab in labels:
        if lab in cr:
            per_label_rows.append({
                "label": lab,
                "precision": cr[lab]["precision"],
                "recall": cr[lab]["recall"],
                "f1": cr[lab]["f1-score"],
                "support": cr[lab]["support"],
            })
    per_label_df = pd.DataFrame(per_label_rows).sort_values("label")
    per_label_df.to_csv(per_label_csv_path, index=False)

    # --- Confusion matrix PNG ---
    _save_confusion_matrix_png(cm, labels, confusion_png_path)

    # Optional: evaluated rows for auditing
    if output_evaluated_rows_path:
        df_out = df[["chatgpt", "gemini", "predicted", "cumulative", "pred_n"]].rename(
            columns={"pred_n": "predicted_normalized"}
        )
        df_out.to_csv(output_evaluated_rows_path, index=False)

    return overall_df, per_label_df


# -----------------------
# Run from VS Code
# -----------------------
if __name__ == "__main__":
    input_path = "12-EVALUATION_QUAL/sentiment_sample_B.csv"

    overall_csv_path = "12-EVALUATION_QUAL/sentiment_results/sentiment_overall.csv"
    per_label_csv_path = "12-EVALUATION_QUAL/sentiment_results/sentiment_label.csv"
    confusion_png_path = "12-EVALUATION_QUAL/sentiment_results/sentiment_confusion_matrix.png"
    evaluated_rows_path = "12-EVALUATION_QUAL/sentiment_results/sentiment_rows.csv"  # optional; set None to disable

    overall_df, per_label_df = evaluate_sentiment_csv(
        input_path=input_path,
        overall_csv_path=overall_csv_path,
        per_label_csv_path=per_label_csv_path,
        confusion_png_path=confusion_png_path,
        output_evaluated_rows_path=evaluated_rows_path,
    )

    print("Saved:", overall_csv_path)
    print("Saved:", per_label_csv_path)
    print("Saved:", confusion_png_path)
    print("Overall metrics:\n", overall_df)
    print("\nPer-label metrics:\n", per_label_df)
