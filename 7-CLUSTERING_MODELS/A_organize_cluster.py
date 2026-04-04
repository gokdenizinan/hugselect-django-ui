import pandas as pd

INPUT_FILE = "7-CLUSTERING_MODELS/clusters_improved/family_assignments_organized.csv"
OUTPUT_FILE = "7-CLUSTERING_MODELS/clusters_improved/organizedd/family_assignments_organized_again.csv"

SORT_ORDER = [
    "assigned_modality",
    "task",
    "family_root",
    "family_child",
    "base_models",
    "library_name",
]

COLUMNS_TO_REMOVE = [
    # "assignment_method",
    # "family_confidence",
    # "candidate_root_raw",
]


def move_after(cols, column_to_move, after_column):
    if column_to_move in cols and after_column in cols:
        cols.remove(column_to_move)
        insert_at = cols.index(after_column) + 1
        cols.insert(insert_at, column_to_move)
    return cols


def make_pair_table(df, lower_col, higher_col, filename):
    table = (
        df.groupby([lower_col, higher_col], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(by=[lower_col, higher_col])
    )
    table.to_csv(filename, index=False)


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Original columns: {df.columns.tolist()}")

    # Remove unwanted columns
    df = df.drop(columns=[c for c in COLUMNS_TO_REMOVE if c in df.columns], errors="ignore")

    # Clean strings
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    # Sort rows (main → smaller clusters)
    missing = [c for c in SORT_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(by=SORT_ORDER, na_position="last").reset_index(drop=True)

    # Reorder columns
    cols = list(df.columns)
    cols = move_after(cols, "task", "assigned_modality")
    cols = move_after(cols, "model_type", "family_root")
    cols = move_after(cols, "baseline_models", "family_child")

    df = df[cols]

    # Save organized CSV
    df.to_csv(OUTPUT_FILE, index=False)

    # === PAIR COUNT TABLES (what you asked for) ===

    make_pair_table(
        df,
        lower_col="family_child",
        higher_col="family_root",
        filename="7-CLUSTERING_MODELS/clusters_improved/organizedd/family_child_to_family_root_counts.csv",
    )

    make_pair_table(
        df,
        lower_col="family_root",
        higher_col="pipeline_tag",
        filename="7-CLUSTERING_MODELS/clusters_improved/organizedd/family_root_to_pipeline_tag_counts.csv",
    )

    make_pair_table(
        df,
        lower_col="pipeline_tag",
        higher_col="assigned_modality",
        filename="7-CLUSTERING_MODELS/clusters_improved/organizedd/pipeline_tag_to_assigned_modality_counts.csv",
    )

    print("Done.")
    print("Created:")
    print("  - organized.csv")
    print("  - family_child_to_family_root_counts.csv")
    print("  - family_root_to_pipeline_tag_counts.csv")
    print("  - pipeline_tag_to_assigned_modality_counts.csv")


if __name__ == "__main__":
    main()