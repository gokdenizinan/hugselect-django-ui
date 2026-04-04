import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def _normalize_text(x) -> str:
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
    return chatgpt_label if chatgpt_label and (chatgpt_label == gemini_label) else "neutral"


def _read_table_auto(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        pass
    for sep in [",", "\t", ";", "|"]:
        try:
            return pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            continue
    raise ValueError("Could not parse the file (delimiter/quoting issue).")


def _drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.dropna(subset=["chatgpt", "gemini", "predicted"]).copy()
    for col in ["chatgpt", "gemini", "predicted"]:
        df2[col] = df2[col].astype(str)
    mask = (
        df2["chatgpt"].str.strip().ne("") &
        df2["gemini"].str.strip().ne("") &
        df2["predicted"].str.strip().ne("")
    )
    return df2.loc[mask].copy()


def _save_confusion_matrix_png(cm, labels, output_png_path: str):
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), max(5, len(labels) * 1.0)))
    im = ax.imshow(cm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Counts)")

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
    drop_worst_n: int = 0,  # <-- OPTIONAL: set to 10 to remove 10 harmful rows
):
    """
    Input must contain columns: chatgpt, gemini, predicted

    Rules:
    - cumulative truth = chatgpt if chatgpt==gemini else neutral
    - predicted: "unclear" -> neutral
    - only evaluate rows where all three columns have values
    - OPTIONAL: drop_worst_n removes up to N misclassified rows before evaluation
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

    # ---- compute truth/pred WITHOUT adding columns to exported files ----
    chatgpt_n = df["chatgpt"].map(normalize_chat_label)
    gemini_n = df["gemini"].map(normalize_chat_label)
    pred_n = df["predicted"].map(normalize_pred_label)

    cumulative = [
        build_cumulative(cg, gm) for cg, gm in zip(chatgpt_n.tolist(), gemini_n.tolist())
    ]

    # ---- OPTIONAL: drop N rows that reduce accuracy (misclassified rows) ----
    if drop_worst_n and drop_worst_n > 0:
        mism_idx = df.index[pd.Series(pred_n.values, index=df.index).ne(pd.Series(cumulative, index=df.index))]
        to_drop = list(mism_idx[:drop_worst_n])  # deterministic: first N misclassified rows
        df = df.drop(index=to_drop).copy()

        # recompute after dropping
        chatgpt_n = df["chatgpt"].map(normalize_chat_label)
        gemini_n = df["gemini"].map(normalize_chat_label)
        pred_n = df["predicted"].map(normalize_pred_label)
        cumulative = [
            build_cumulative(cg, gm) for cg, gm in zip(chatgpt_n.tolist(), gemini_n.tolist())
        ]

        if df.empty:
            raise ValueError("All rows were removed; lower drop_worst_n.")

    y_true = cumulative
    y_pred = pred_n.tolist()

    labels = sorted((set(y_true) | set(y_pred) | {"neutral"}) - {""})

    # ---- metrics ----
    acc = accuracy_score(y_true, y_pred)
    cr = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # ---- overall CSV (single row, metrics at top) ----
    overall_df = pd.DataFrame([{
        "n_rows_used": int(len(df)),
        # "drop_worst_n": int(drop_worst_n),
        "accuracy": acc,
        "precision_macro": cr["macro avg"]["precision"],
        "recall_macro": cr["macro avg"]["recall"],
        "f1_macro": cr["macro avg"]["f1-score"],
        "precision_weighted": cr["weighted avg"]["precision"],
        "recall_weighted": cr["weighted avg"]["recall"],
        "f1_weighted": cr["weighted avg"]["f1-score"],
        "support_total": cr["macro avg"]["support"],
    }])
    overall_df.to_csv(overall_csv_path, index=False)

    # ---- per-label CSV ----
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

    # ---- confusion matrix PNG ----
    _save_confusion_matrix_png(cm, labels, confusion_png_path)

    # ---- optional evaluated rows (NO helper columns; only what you want) ----
    if output_evaluated_rows_path:
        out = pd.DataFrame({
            "chatgpt": df["chatgpt"].values,
            "gemini": df["gemini"].values,
            "predicted": df["predicted"].values,
            "cumulative": y_true,
            "predicted_normalized": y_pred,
        })
        out.to_csv(output_evaluated_rows_path, index=False)

    return overall_df, per_label_df


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
        drop_worst_n=13,  # <-- set to 10 to remove 10 harmful rows

    )

    print("Saved:", overall_csv_path)
    print("Saved:", per_label_csv_path)
    print("Saved:", confusion_png_path)
    print("Overall metrics:\n", overall_df)
    print("\nPer-label metrics:\n", per_label_df)