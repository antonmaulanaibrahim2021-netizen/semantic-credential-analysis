"""
LABEL CONSTRUCTION AUDIT
Q1: Semantic Consistency Assessment of Digital Academic Credentials
Using Machine Learning

Purpose:
    Audit whether the target label can be deterministically reconstructed
    from combinations of credential attributes.

This script DOES NOT modify the dataset.

It evaluates:
    1. Single categorical features
    2. Two-feature combinations
    3. Three-feature combinations
    4. Four-feature combinations
    5. All available categorical features

For each combination, the script identifies:
    - Number of unique combinations
    - Combinations containing only label 0
    - Combinations containing only label 1
    - Combinations containing both labels
    - Deterministic percentage
    - Conflicting percentage

IMPORTANT:
    A deterministic relationship does NOT automatically mean target leakage.
    It only indicates that the observed label can be reconstructed from
    the selected attributes. The data-generation process must be examined
    before calling this leakage.
"""

from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "cross_institutional_dataset (1).xlsx"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = (
    RESULTS_DIR / "label_construction_audit_report.txt"
)

SUMMARY_FILE = (
    RESULTS_DIR / "label_construction_summary.csv"
)

DETAIL_FILE = (
    RESULTS_DIR / "label_construction_details.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TARGET_COLUMN = "label"

# Categorical credential attributes identified during data audit.
CATEGORICAL_FEATURES = [
    "institution",
    "program",
    "education_level",
    "accreditation_status",
    "graduation_predicate",
    "faculty_program_name",
]


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset():

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            "\nDataset tidak ditemukan:\n"
            f"{DATASET_FILE}\n\n"
            "Pastikan file berada di folder dataset."
        )

    print("\nLoading dataset...")

    df = pd.read_excel(
        DATASET_FILE,
        sheet_name="Combined_Labeled",
        skiprows=2
    )

    # Clean column names only.
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ============================================================
# 4. CHECK DATASET STRUCTURE
# ============================================================

def check_dataset(df):

    print("\n=== DATASET CHECK ===")

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"\nKolom target '{TARGET_COLUMN}' tidak ditemukan."
        )

    print("\nLabel distribution:")

    label_distribution = (
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
    )

    print(label_distribution)


# ============================================================
# 5. HANDLE MISSING VALUES FOR GROUPING
# ============================================================

def prepare_features(df, features):

    temp = df[features + [TARGET_COLUMN]].copy()

    # Missing values are explicitly represented.
    # This allows us to evaluate whether missingness itself
    # contributes to deterministic label patterns.
    for column in features:

        temp[column] = (
            temp[column]
            .astype("object")
            .where(
                temp[column].notna(),
                "__MISSING__"
            )
        )

    return temp


# ============================================================
# 6. ANALYZE ONE FEATURE COMBINATION
# ============================================================

def analyze_combination(df, features):

    temp = prepare_features(
        df,
        features
    )

    # Group observations according to selected features.
    grouped = (
        temp
        .groupby(
            features,
            dropna=False
        )[TARGET_COLUMN]
        .agg(
            unique_labels="nunique",
            count="count"
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Classification of groups
    # --------------------------------------------------------

    deterministic_groups = grouped[
        grouped["unique_labels"] == 1
    ]

    conflicting_groups = grouped[
        grouped["unique_labels"] > 1
    ]

    total_groups = len(grouped)

    deterministic_group_count = len(
        deterministic_groups
    )

    conflicting_group_count = len(
        conflicting_groups
    )

    if total_groups > 0:

        deterministic_percentage = (
            deterministic_group_count
            / total_groups
            * 100
        )

        conflicting_percentage = (
            conflicting_group_count
            / total_groups
            * 100
        )

    else:

        deterministic_percentage = 0
        conflicting_percentage = 0

    # --------------------------------------------------------
    # Observation-level coverage
    # --------------------------------------------------------

    deterministic_observations = (
        deterministic_groups["count"].sum()
    )

    conflicting_observations = (
        conflicting_groups["count"].sum()
    )

    total_observations = len(temp)

    if total_observations > 0:

        deterministic_observation_percentage = (
            deterministic_observations
            / total_observations
            * 100
        )

        conflicting_observation_percentage = (
            conflicting_observations
            / total_observations
            * 100
        )

    else:

        deterministic_observation_percentage = 0
        conflicting_observation_percentage = 0

    # --------------------------------------------------------
    # Majority-rule accuracy
    # --------------------------------------------------------

    # For every feature combination, use the majority label.
    # This provides a simple descriptive measure of how much
    # label information is captured by the combination.

    grouped_label_counts = (
        temp
        .groupby(
            features + [TARGET_COLUMN],
            dropna=False
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    majority_counts = (
        grouped_label_counts
        .groupby(features, dropna=False)["count"]
        .max()
    )

    majority_accuracy = (
        majority_counts.sum()
        / total_observations
        * 100
        if total_observations > 0
        else 0
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "features": " + ".join(features),
        "n_features": len(features),
        "unique_combinations": total_groups,

        "deterministic_groups": deterministic_group_count,
        "conflicting_groups": conflicting_group_count,

        "deterministic_group_percentage":
            round(
                deterministic_percentage,
                4
            ),

        "conflicting_group_percentage":
            round(
                conflicting_percentage,
                4
            ),

        "deterministic_observations":
            int(deterministic_observations),

        "conflicting_observations":
            int(conflicting_observations),

        "deterministic_observation_percentage":
            round(
                deterministic_observation_percentage,
                4
            ),

        "conflicting_observation_percentage":
            round(
                conflicting_observation_percentage,
                4
            ),

        "majority_rule_accuracy":
            round(
                majority_accuracy,
                4
            ),
    }


# ============================================================
# 7. GENERATE FEATURE COMBINATIONS
# ============================================================

def generate_combinations(features):

    all_combinations = []

    # Single feature
    for feature in features:

        all_combinations.append(
            (feature,)
        )

    # Pair, triple, etc.
    for r in range(2, len(features) + 1):

        for combination in combinations(
            features,
            r
        ):

            all_combinations.append(
                combination
            )

    return all_combinations


# ============================================================
# 8. DETAILED ANALYSIS OF IMPORTANT COMBINATIONS
# ============================================================

def generate_detail_table(
    df,
    feature_combinations
):

    detail_rows = []

    for features in feature_combinations:

        temp = prepare_features(
            df,
            list(features)
        )

        grouped = (
            temp
            .groupby(
                list(features) + [TARGET_COLUMN],
                dropna=False
            )
            .size()
            .reset_index(
                name="count"
            )
        )

        # Create a feature-combination identifier.
        grouped["feature_combination"] = (
            " + ".join(features)
        )

        detail_rows.append(
            grouped
        )

    if detail_rows:

        detail_df = pd.concat(
            detail_rows,
            ignore_index=True
        )

    else:

        detail_df = pd.DataFrame()

    return detail_df


# ============================================================
# 9. PRINT RESULTS
# ============================================================

def print_summary(summary_df):

    print("\n")
    print("=" * 70)
    print("LABEL CONSTRUCTION AUDIT RESULTS")
    print("=" * 70)

    # Sort by majority-rule accuracy.
    display_df = (
        summary_df
        .sort_values(
            "majority_rule_accuracy",
            ascending=False
        )
    )

    print(
        display_df[
            [
                "features",
                "n_features",
                "unique_combinations",
                "deterministic_groups",
                "conflicting_groups",
                "deterministic_observation_percentage",
                "majority_rule_accuracy",
            ]
        ].to_string(index=False)
    )


# ============================================================
# 10. PRELIMINARY INTERPRETATION
# ============================================================

def generate_interpretation(
    summary_df
):

    report = []

    report.append(
        "\nPRELIMINARY INTERPRETATION"
    )

    report.append(
        "-" * 70
    )

    report.append(
        "This audit identifies deterministic relationships between "
        "credential attributes and the observed target label."
    )

    report.append(
        "A deterministic relationship does not automatically constitute "
        "target leakage. Leakage requires evidence that information "
        "derived from the target or future information entered the "
        "predictor variables or the modeling process."
    )

    # Best descriptive combination.
    best_row = (
        summary_df
        .sort_values(
            "majority_rule_accuracy",
            ascending=False
        )
        .iloc[0]
    )

    report.append(
        "\nHighest majority-rule accuracy:"
    )

    report.append(
        f"- Features: {best_row['features']}"
    )

    report.append(
        f"- Accuracy: "
        f"{best_row['majority_rule_accuracy']:.4f}%"
    )

    report.append(
        f"- Unique combinations: "
        f"{best_row['unique_combinations']}"
    )

    report.append(
        f"- Deterministic groups: "
        f"{best_row['deterministic_groups']}"
    )

    report.append(
        f"- Conflicting groups: "
        f"{best_row['conflicting_groups']}"
    )

    report.append(
        f"- Deterministic observation coverage: "
        f"{best_row['deterministic_observation_percentage']:.4f}%"
    )

    report.append(
        "\nInterpretation rule:"
    )

    report.append(
        "If a combination has 100% majority-rule accuracy and "
        "zero conflicting groups, the observed label can be "
        "reconstructed from that combination in the current dataset."
    )

    report.append(
        "This finding should be investigated against the original "
        "label-generation procedure before being interpreted as "
        "evidence of semantic validity or machine-learning capability."
    )

    report.append(
        "\nThe audit does not modify, remove, or relabel any record."
    )

    return report


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Q1 SEMANTIC CREDENTIAL RESEARCH")
    print("LABEL CONSTRUCTION AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    check_dataset(df)

    # --------------------------------------------------------
    # Identify available categorical features
    # --------------------------------------------------------

    available_features = [
        feature
        for feature in CATEGORICAL_FEATURES
        if feature in df.columns
    ]

    print("\nAvailable categorical features:")

    for feature in available_features:
        print(f"- {feature}")

    # --------------------------------------------------------
    # Generate combinations
    # --------------------------------------------------------

    feature_combinations = generate_combinations(
        available_features
    )

    print(
        f"\nTotal feature combinations tested: "
        f"{len(feature_combinations)}"
    )

    # --------------------------------------------------------
    # Analyze combinations
    # --------------------------------------------------------

    results = []

    for index, features in enumerate(
        feature_combinations,
        start=1
    ):

        print(
            f"Analyzing "
            f"{index}/{len(feature_combinations)}: "
            f"{' + '.join(features)}"
        )

        result = analyze_combination(
            df,
            list(features)
        )

        results.append(result)

    summary_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Detailed table
    # --------------------------------------------------------

    detail_df = generate_detail_table(
        df,
        feature_combinations
    )

    detail_df.to_csv(
        DETAIL_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_summary(
        summary_df
    )

    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    report = []

    report.append(
        "=" * 70
    )

    report.append(
        "LABEL CONSTRUCTION AUDIT REPORT"
    )

    report.append(
        "=" * 70
    )

    report.append(
        f"\nDataset: {DATASET_FILE.name}"
    )

    report.append(
        f"Rows: {len(df)}"
    )

    report.append(
        f"Columns: {len(df.columns)}"
    )

    report.append(
        f"Target: {TARGET_COLUMN}"
    )

    report.append(
        "\nTarget distribution:"
    )

    report.append(
        df[TARGET_COLUMN]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_string()
    )

    report.append(
        "\nCategorical features analyzed:"
    )

    for feature in available_features:

        report.append(
            f"- {feature}"
        )

    report.append(
        f"\nNumber of feature combinations tested: "
        f"{len(feature_combinations)}"
    )

    report.append(
        "\nFULL SUMMARY"
    )

    report.append(
        "-" * 70
    )

    report.append(
        summary_df
        .sort_values(
            "majority_rule_accuracy",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    report.extend(
        generate_interpretation(
            summary_df
        )
    )

    report.append(
        "\nOUTPUT FILES"
    )

    report.append(
        "-" * 70
    )

    report.append(
        f"Summary : {SUMMARY_FILE}"
    )

    report.append(
        f"Details : {DETAIL_FILE}"
    )

    report.append(
        f"Report  : {REPORT_FILE}"
    )

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("LABEL CONSTRUCTION AUDIT SELESAI")
    print("=" * 70)

    print(
        f"\nReport saved to:"
    )

    print(
        REPORT_FILE
    )

    print(
        f"\nSummary saved to:"
    )

    print(
        SUMMARY_FILE
    )

    print(
        f"\nDetails saved to:"
    )

    print(
        DETAIL_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not interpret deterministic relationships as "
        "target leakage automatically."
    )

    print(
        "Review the original label-generation process first."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()