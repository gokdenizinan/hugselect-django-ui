import pandas as pd
import ast
import re

def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

STOPWORDS = {
    "model", "models", "file", "files", "format", "type", "types",
    "data", "system", "framework"
}

# OLD MATCHING CRITERIA
def fuzzy_match(pred: str, act: str) -> bool:
    p = normalize(pred)
    a = normalize(act)

    if not p or not a:
        return False

    p_tokens = set(p.split())
    a_tokens = set(a.split())

    # Remove generic tokens
    p_core = p_tokens - STOPWORDS
    a_core = a_tokens - STOPWORDS
    # EXACT MATCH
    # return p_core == a_core


    # Strong signal: any shared core token
    
    if len(p_core & a_core) >= 1:
        return True


    # Fallback: abbreviation containment (GGUF-style)
    for t in p_tokens:
        if len(t) >= 3 and t in a:
            return True
    for t in a_tokens:
        if len(t) >= 3 and t in p:
            return True

    return False

def compute_effective_label_size(df, actual_col, predicted_col):
    all_labels = []

    # Collect all labels
    for _, row in df.iterrows():
        actual = ensure_list(row[actual_col])
        predicted = ensure_list(row[predicted_col])
        all_labels.extend(actual)
        all_labels.extend(predicted)

    # Normalize first (important for stability)
    all_labels = [normalize(label) for label in all_labels if label]

    unique_labels = list(set(all_labels))

    clusters = []
    visited = set()

    for label in unique_labels:
        if label in visited:
            continue

        cluster = {label}
        visited.add(label)

        for other in unique_labels:
            if other in visited:
                continue

            if fuzzy_match(label, other):
                cluster.add(other)
                visited.add(other)

        clusters.append(cluster)

    return len(clusters)


def ensure_list(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        # stringified list like "['Stanza']" or '["Stanza"]'
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                return v if isinstance(v, list) else [v]
            except Exception:
                pass
        # allow single label strings
        return [s] if s else []
    try:
        return list(x)
    except TypeError:
        return [x]

def find_matches(actual, predicted):
    matches = []
    for p in predicted:
        for a in actual:
            if fuzzy_match(p, a):
                matches.append((p, a))
    return matches

def evaluate_list_labels(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    print_matches: bool = True
):
    total_rows = len(df)
    hit_rows = 0

    total_predicted = 0
    total_actual = 0
    total_correct = 0

    for idx, row in df.iterrows():
        actual = set(ensure_list(row[actual_col]))
        predicted = set(ensure_list(row[predicted_col]))

        total_predicted += len(predicted)
        total_actual += len(actual)

        matches = find_matches(actual, predicted)
        correct_preds = {p for p, _ in matches}

        total_correct += len(correct_preds)

        if matches:
            hit_rows += 1

            if print_matches:
                print(f"\nRow {idx}")
                # print(f"  Actual   : {actual}")
                # print(f"  Predicted: {predicted}")
                print("  Matches:")
                for p, a in matches:
                    print(f"    ✓ '{p}'  ↔  '{a}'")

    accuracy = hit_rows / total_rows if total_rows else 0
    precision = total_correct / total_predicted if total_predicted else 0
    recall = total_correct / total_actual if total_actual else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    return {
        "accuracy_row_level": accuracy,
        "precision_label_level": precision,
        "recall_label_level": recall,
        "f1_label_level": f1
    }


def exact_subset_accuracy(df: pd.DataFrame, actual_col: str, predicted_col: str):
    """Row is correct if all predicted labels are contained in actual labels (exact string match)."""
    correct = 0
    for _, row in df.iterrows():
        actual = set(ensure_list(row[actual_col]))
        predicted = set(ensure_list(row[predicted_col]))
        if predicted.issubset(actual):
            correct += 1
    return correct / len(df) if len(df) else 0

def jaccard_score(df: pd.DataFrame, actual_col: str, predicted_col: str):
    scores = []
    for _, row in df.iterrows():
        actual = set(ensure_list(row[actual_col]))
        predicted = set(ensure_list(row[predicted_col]))
        union = len(actual | predicted)
        scores.append(len(actual & predicted) / union if union else 0)
    return sum(scores) / len(scores) if len(scores) else 0

def evaluate_overlap_metrics(df, actual_col, predicted_col):
    TP = FP = FN = 0

    for _, row in df.iterrows():
        actual = set(ensure_list(row[actual_col]))
        predicted = set(ensure_list(row[predicted_col]))

        matched_actual = set()
        matched_predicted = set()

        for p in predicted:
            for a in actual:
                if fuzzy_match(p, a):
                    matched_predicted.add(p)
                    matched_actual.add(a)

        TP += len(matched_predicted)
        FP += len(predicted - matched_predicted)
        FN += len(actual - matched_actual)
        

    total_labels = compute_effective_label_size(df, actual_col, predicted_col)
    total_decisions = total_labels * len(df)
    TN = total_decisions - (TP + FP + FN)

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    overlap_jaccard = TP / (TP + FP + FN) if (TP + FP + FN) else 0
    overlap_accuracy = (TP + TN) / total_decisions if total_decisions else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    return {
        "overlap_accuracy": overlap_accuracy,
        "overlap_jaccard": overlap_jaccard,  # same as accuracy in this definition
        "overlap_precision": precision,
        "overlap_recall": recall,
        "overlap_f1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN,
    }

def add_row_accuracy_and_save(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    output_csv: str,
    acc_col: str = "row_accuracy",
    add_counts: bool = True,
) -> pd.DataFrame:
    """
    Row accuracy = Jaccard/IoU = TP / (TP + FP + FN), using your matching rule:
      - TP: predicted item counts as TP if it matches ANY actual item (fuzzy_match)
      - FP: predicted items with no match
      - FN: actual items with no predicted match
    """

    def compute_row(actual_cell, predicted_cell):
        actual = set(ensure_list(actual_cell))
        predicted = set(ensure_list(predicted_cell))

        matched_pred = set()
        matched_act = set()

        for p in predicted:
            for a in actual:
                if fuzzy_match(p, a):
                    matched_pred.add(p)   # pred-based TP
                    matched_act.add(a)    # for FN
                    break

        TP = len(matched_pred)
        FP = len(predicted) - TP
        FN = len(actual) - len(matched_act)

        denom = TP + FP + FN
        acc = (TP / denom) if denom else 0.0  # ALWAYS 0..1

        return acc, TP, FP, FN

    out = df.copy()
    rows = out.apply(lambda r: compute_row(r[actual_col], r[predicted_col]), axis=1)

    out[acc_col] = [float(x[0]) for x in rows]  # force float
    out[acc_col] = out[acc_col].map(lambda x: f"{x:.3f}")


    if add_counts:
        out["row_TP"] = [x[1] for x in rows]
        out["row_FP"] = [x[2] for x in rows]
        out["row_FN"] = [x[3] for x in rows]

    out.to_csv(output_csv, index=False)
    return out

def add_row_accuracy_with_examples_and_save(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    output_csv: str,
    acc_col: str = "row_accuracy",
) -> pd.DataFrame:
    """
    Row accuracy = Jaccard / IoU = TP / (TP + FP + FN)

    TP rule:
      - Each predicted item counts as TP if it fuzzy-matches ANY actual item.
      - FP = predicted items with no match
      - FN = actual items with no predicted match

    Adds example columns:
      - tp_examples
      - fp_examples
      - fn_examples
    """

    def compute_row(actual_cell, predicted_cell):
        actual = set(ensure_list(actual_cell))
        predicted = set(ensure_list(predicted_cell))

        matched_pred = set()
        matched_act = set()

        for p in predicted:
            for a in actual:
                if fuzzy_match(p, a):
                    matched_pred.add(p)
                    matched_act.add(a)
                    break  # pred-based TP

        TP = len(matched_pred)
        FP = len(predicted) - TP
        FN = len(actual) - len(matched_act)

        denom = TP + FP + FN
        acc = TP / denom if denom else 0.0

        tp_examples = sorted(matched_pred)
        fp_examples = sorted(predicted - matched_pred)
        fn_examples = sorted(actual - matched_act)

        return acc, TP, FP, FN, tp_examples, fp_examples, fn_examples

    out = df.copy()
    rows = out.apply(
        lambda r: compute_row(r[actual_col], r[predicted_col]),
        axis=1
    )

    out[acc_col] = [float(r[0]) for r in rows]
    out[acc_col] = out[acc_col].map(lambda x: f"{x:.3f}")
    
    out["row_TP"] = [r[1] for r in rows]
    out["row_FP"] = [r[2] for r in rows]
    out["row_FN"] = [r[3] for r in rows]

    out["tp_examples"] = [r[4] for r in rows]
    out["fp_examples"] = [r[5] for r in rows]
    out["fn_examples"] = [r[6] for r in rows]

    out.to_csv(output_csv, index=False)
    return out


CSV_IN = "10-EVALUATION/models_ffs/model_ffs_eval_T2.csv"
CSV_OUT = "10-EVALUATION/models_ffs/model_ffs_eval_T2_relax.csv"

df = pd.read_csv(CSV_IN)

df = add_row_accuracy_with_examples_and_save(
    df,
    actual_col="truth",
    predicted_col="Features",
    output_csv= CSV_OUT)

metrics_overlap = evaluate_overlap_metrics(df, "truth", "Features")

print("OVERLAP METRICS (global, feature-level)")
print("Accuracy:", metrics_overlap["overlap_accuracy"])
print("Jaccard:", metrics_overlap["overlap_jaccard"])
print("Precision:", metrics_overlap["overlap_precision"])
print("Recall:", metrics_overlap["overlap_recall"])
print("F1:", metrics_overlap["overlap_f1"])


