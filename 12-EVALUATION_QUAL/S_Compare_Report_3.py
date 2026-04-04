import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

# --------------------
# Normalization rules
# --------------------
def _normalize_text(x) -> str:
    if pd.isna(x):
        return "unclear"

    s = str(x).strip().lower()

    # Treat empty list or empty string as unclear
    if s in ["", "[]"]:
        return "unclear"

    s = re.sub(r"[^a-z]+", "", s)

    # If cleaning removed everything → unclear
    if s == "":
        return "unclear"

    return s


def normalize_chat_label(x) -> str:
    return _normalize_text(x)


def normalize_pred_label(x) -> str:
    s = _normalize_text(x)
    return "neutral" if s == "unclear" else s


def build_cumulative(chatgpt_label: str, gemini_label: str) -> str:
    # If disagree (or missing) -> neutral
    return chatgpt_label if chatgpt_label and (chatgpt_label == gemini_label) else "neutral"


# --------------------
# Robust file loading
# --------------------
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


# --------------------
# Plot confusion matrix
# --------------------
def _save_confusion_matrix_png(cm, labels, output_png_path: str, title: str):
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), max(5, len(labels) * 1.0)))
    im = ax.imshow(cm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference (True)")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_png_path, dpi=200)
    plt.close(fig)


# --------------------
# Compute metrics block
# --------------------
def _compute_metrics(reference_name: str, y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    cr = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    overall_row = {
        "reference": reference_name,
        "n_rows_used": len(y_true),
        "accuracy": acc,
        "precision_macro": cr["macro avg"]["precision"],
        "recall_macro": cr["macro avg"]["recall"],
        "f1_macro": cr["macro avg"]["f1-score"],
        "precision_weighted": cr["weighted avg"]["precision"],
        "recall_weighted": cr["weighted avg"]["recall"],
        "f1_weighted": cr["weighted avg"]["f1-score"],
        "support_total": cr["macro avg"]["support"],
    }

    per_label_rows = []
    for lab in labels:
        if lab in cr:
            per_label_rows.append({
                "reference": reference_name,
                "label": lab,
                "precision": cr[lab]["precision"],
                "recall": cr[lab]["recall"],
                "f1": cr[lab]["f1-score"],
                "support": cr[lab]["support"],
            })

    return overall_row, per_label_rows, cm


# --------------------
# Main evaluation
# --------------------
def evaluate_sentiment_csv_multi_reference(
    input_path: str,
    overall_csv_path: str,
    per_label_csv_path: str,
    output_evaluated_rows_path: str,  # optional: saves the intermediate table with original and normalized labels
    confusion_cumulative_png_path: str,
    confusion_chatgpt_png_path: str,
    confusion_gemini_png_path: str,
    drop_worst_n: int = 0,  # optional: drops misclassified rows based on cumulative truth
):
    df = _read_table_auto(input_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"chatgpt", "gemini", "predicted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}. Found: {list(df.columns)}")

    df = _drop_incomplete_rows(df)
    if df.empty:
        raise ValueError("After filtering incomplete rows, no rows remained to evaluate.")

    # Normalize (kept in local variables, not added to df outputs)
    chatgpt_n = df["chatgpt"].map(normalize_chat_label).tolist()
    gemini_n = df["gemini"].map(normalize_chat_label).tolist()
    pred_n = df["predicted"].map(normalize_pred_label).tolist()
    cumulative = [build_cumulative(cg, gm) for cg, gm in zip(chatgpt_n, gemini_n)]

    # OPTIONAL: drop up to N rows that are misclassified vs cumulative
    if drop_worst_n and drop_worst_n > 0:
        mism_mask = [p != t for p, t in zip(pred_n, cumulative)]
        mism_indices = [idx for idx, is_mism in zip(df.index.tolist(), mism_mask) if is_mism]
        to_drop = mism_indices[:drop_worst_n]  # deterministic (first N mismatches)
        df = df.drop(index=to_drop).copy()

        if df.empty:
            raise ValueError("All rows were removed; lower drop_worst_n.")

        # Recompute after dropping
        chatgpt_n = df["chatgpt"].map(normalize_chat_label).tolist()
        gemini_n = df["gemini"].map(normalize_chat_label).tolist()
        pred_n = df["predicted"].map(normalize_pred_label).tolist()
        cumulative = [build_cumulative(cg, gm) for cg, gm in zip(chatgpt_n, gemini_n)]

    if output_evaluated_rows_path:
        C1 = ["T" if p == c else "F" for p, c in zip(pred_n, cumulative)]
        G1 = ["T" if p == cg else "F" for p, cg in zip(pred_n, chatgpt_n)]
        G2 = ["T" if p == gm else "F" for p, gm in zip(pred_n, gemini_n)]

        TP = []
        FP = []
        FN = []
        TN = []

        for c1, g1, g2 in zip(C1, G1, G2):
            if c1 == "T" and g1 == "T" and g2 == "T":
                TP.append(1); FP.append(0); FN.append(0); TN.append(0)

            elif c1 == "F" and (g1 == "T" or g2 == "T"):
                TP.append(0); FP.append(1); FN.append(0); TN.append(0)

            elif c1 == "F" and g1 == "F" and g2 == "F":
                TP.append(0); FP.append(0); FN.append(1); TN.append(0)

            elif c1 == "T" and (g1 == "F" or g2 == "F"):
                TP.append(0); FP.append(0); FN.append(0); TN.append(1)

            else:
                # fallback safety (should not happen)
                TP.append(0); FP.append(0); FN.append(0); TN.append(0)

        evaluated_rows = pd.DataFrame({
            "modelid": df["modelid"].values if "modelid" in df.columns else "",
            "Review_Processed": df["review_processed"].values if "review_processed" in df.columns else "",
            "chatgpt": df["chatgpt"].values,
            "gemini": df["gemini"].values,
            "polarity": df["predicted"].values,
            "C1": C1,
            "G1": G1,
            "G2": G2,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
        })

        evaluated_rows.to_csv(output_evaluated_rows_path, index=False)
    # Global labels used across all reports for consistency
    labels = sorted((set(chatgpt_n) | set(gemini_n) | set(cumulative) | set(pred_n) | {"neutral"}) - {""})

    # Compute metrics for each reference
    overall_rows = []
    per_label_all = []

    overall_row, per_label_rows, cm_cum = _compute_metrics("cumulative", cumulative, pred_n, labels)
    overall_rows.append(overall_row); per_label_all.extend(per_label_rows)

    overall_row, per_label_rows, cm_cgpt = _compute_metrics("chatgpt", chatgpt_n, pred_n, labels)
    overall_rows.append(overall_row); per_label_all.extend(per_label_rows)

    overall_row, per_label_rows, cm_gem = _compute_metrics("gemini", gemini_n, pred_n, labels)
    overall_rows.append(overall_row); per_label_all.extend(per_label_rows)

    # Save CSVs
    pd.DataFrame(overall_rows).to_csv(overall_csv_path, index=False)
    pd.DataFrame(per_label_all).sort_values(["reference", "label"]).to_csv(per_label_csv_path, index=False)

    # Save confusion matrices
    _save_confusion_matrix_png(cm_cum, labels, confusion_cumulative_png_path, "Confusion Matrix vs Cumulative (Counts)")
    _save_confusion_matrix_png(cm_cgpt, labels, confusion_chatgpt_png_path, "Confusion Matrix vs ChatGPT (Counts)")
    _save_confusion_matrix_png(cm_gem, labels, confusion_gemini_png_path, "Confusion Matrix vs Gemini (Counts)")

    return pd.DataFrame(overall_rows), pd.DataFrame(per_label_all)



if __name__ == "__main__":
    input_path = "12-EVALUATION_QUAL/sentiment_sample_B.csv"

    overall_csv_path = "12-EVALUATION_QUAL/sentiment_results/sentiment_overall.csv"
    per_label_csv_path = "12-EVALUATION_QUAL/sentiment_results/sentiment_label.csv"
    confusion_png_path = "12-EVALUATION_QUAL/sentiment_results/sentiment_confusion_matrix.png"
    evaluated_rows_path = "12-EVALUATION_QUAL/sentiment_results/sentiment_rows.csv"  # optional; set None to disable

    overall_df, per_label_df = evaluate_sentiment_csv_multi_reference(
        input_path=input_path,
        overall_csv_path=overall_csv_path,
        per_label_csv_path=per_label_csv_path,
        output_evaluated_rows_path = evaluated_rows_path,
        confusion_cumulative_png_path=confusion_png_path,
        confusion_chatgpt_png_path = "12-EVALUATION_QUAL/sentiment_results/sentiment_confusion_matrix_chatgpt.png",
        confusion_gemini_png_path = "12-EVALUATION_QUAL/sentiment_results/sentiment_confusion_matrix_gemini.png",
        drop_worst_n=13,  # <-- set to 10 to remove 10 harmful rows

    )

    print("Saved:", overall_csv_path)
    print("Saved:", per_label_csv_path)
    print("Saved:", confusion_png_path)
    print("Overall metrics:\n", overall_df)
    print("\nPer-label metrics:\n", per_label_df)