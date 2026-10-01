"""
MANIPULATION TYPE AUDIT
Q1: Semantic Consistency Assessment of Digital Academic Credentials
Using Machine Learning

Purpose
-------
Audit the relationship between `manipulation_type` and the target
`label`, and investigate whether manipulation_type may act as:

1. A direct proxy for the target label.
2. A deterministic predictor of the target.
3. A source variable used to construct the label.
4. A variable associated with changes in semantic credential attributes.

IMPORTANT
---------
This script DOES NOT:
- modify the dataset
- delete rows
- change labels
- train the final ML models
- claim leakage automatically

A strong association between manipulation_type and label is not,
by itself, sufficient evidence of target leakage. The original
dataset-generation procedure must also be examined.
"""

from pathlib import Path

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
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE = (
    RESULTS_DIR
    / "manipulation_type_audit_report.txt"
)

DISTRIBUTION_FILE = (
    RESULTS_DIR
    / "manipulation_type_distribution.csv"
)

CROSSTAB_FILE = (
    RESULTS_DIR
    / "manipulation_type_label_crosstab.csv"
)

ATTRIBUTE_CROSSTAB_FILE = (
    RESULTS_DIR
    / "manipulation_type_attribute_relationships.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TARGET_COLUMN = "label"

MANIPULATION_COLUMN = "manipulation_type"

SEMANTIC_ATTRIBUTES = [
    "institution",
    "program",
    "education_level",
    "accreditation_status",
    "gpa",
    "graduation_predicate",
    "graduation_date",
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
# 4. DATASET CHECK
# ============================================================

def check_required_columns(df):

    required_columns = [
        TARGET_COLUMN,
        MANIPULATION_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nKolom berikut tidak ditemukan:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )


# ============================================================
# 5. BASIC INFORMATION
# ============================================================

def print_basic_information(df):

    print("\n")
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    print("\nTarget distribution:")

    print(
        df[TARGET_COLUMN]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print(
        "\nManipulation type distribution:"
    )

    print(
        df[MANIPULATION_COLUMN]
        .value_counts(
            dropna=False
        )
    )


# ============================================================
# 6. MANIPULATION TYPE DISTRIBUTION
# ============================================================

def analyze_manipulation_distribution(df):

    counts = (
        df[MANIPULATION_COLUMN]
        .value_counts(
            dropna=False
        )
        .rename("count")
    )

    percentages = (
        counts
        / len(df)
        * 100
    )

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(4)
    })

    result.index.name = MANIPULATION_COLUMN

    result.to_csv(
        DISTRIBUTION_FILE
    )

    return result


# ============================================================
# 7. MANIPULATION TYPE VS LABEL
# ============================================================

def analyze_manipulation_vs_label(df):

    print("\n")
    print("=" * 70)
    print("MANIPULATION TYPE VS LABEL")
    print("=" * 70)

    crosstab = pd.crosstab(
        df[MANIPULATION_COLUMN],
        df[TARGET_COLUMN],
        dropna=False
    )

    # Make sure label columns 0 and 1 exist.
    for label_value in [0, 1]:

        if label_value not in crosstab.columns:

            crosstab[label_value] = 0

    crosstab = crosstab[
        [0, 1]
    ]

    print("\nCounts:")
    print(crosstab)

    # Row percentages.
    row_percentages = (
        crosstab
        .div(
            crosstab.sum(axis=1),
            axis=0
        )
        * 100
    )

    row_percentages = row_percentages.round(4)

    print("\nRow percentages:")
    print(row_percentages)

    # Save count table.
    crosstab.to_csv(
        CROSSTAB_FILE
    )

    # --------------------------------------------------------
    # Deterministic manipulation categories
    # --------------------------------------------------------

    deterministic_categories = []

    for manipulation_type, row in crosstab.iterrows():

        total = row.sum()

        if total == 0:
            continue

        non_zero_labels = (
            row[row > 0]
        )

        if len(non_zero_labels) == 1:

            deterministic_categories.append(
                {
                    "manipulation_type":
                        manipulation_type,
                    "label":
                        int(non_zero_labels.index[0]),
                    "count":
                        int(total),
                    "percentage":
                        round(
                            100 * total / len(df),
                            4
                        )
                }
            )

    deterministic_df = pd.DataFrame(
        deterministic_categories
    )

    return (
        crosstab,
        row_percentages,
        deterministic_df
    )


# ============================================================
# 8. MAJORITY LABEL ANALYSIS
# ============================================================

def analyze_majority_label(
    crosstab,
    df
):

    print("\n")
    print("=" * 70)
    print("MAJORITY LABEL ANALYSIS")
    print("=" * 70)

    majority_predictions = (
        crosstab
        .idxmax(axis=1)
    )

    category_counts = (
        crosstab.sum(axis=1)
    )

    correctly_classified = (
        crosstab
        .max(axis=1)
    )

    majority_accuracy = (
        correctly_classified.sum()
        / len(df)
        * 100
    )

    print(
        "\nMajority-rule accuracy using "
        "manipulation_type only:"
    )

    print(
        f"{majority_accuracy:.4f}%"
    )

    majority_table = pd.DataFrame({
        "majority_label":
            majority_predictions.astype(int),
        "category_count":
            category_counts.astype(int),
        "correct_by_majority":
            correctly_classified.astype(int)
    })

    majority_table["accuracy_within_category"] = (
        majority_table["correct_by_majority"]
        / majority_table["category_count"]
        * 100
    ).round(4)

    print(
        "\nCategory-level majority analysis:"
    )

    print(
        majority_table.to_string()
    )

    return (
        majority_accuracy,
        majority_table
    )


# ============================================================
# 9. MANIPULATION TYPE VS SEMANTIC ATTRIBUTES
# ============================================================

def analyze_attribute_relationships(df):

    print("\n")
    print("=" * 70)
    print("MANIPULATION TYPE VS SEMANTIC ATTRIBUTES")
    print("=" * 70)

    results = []

    available_attributes = [
        column
        for column in SEMANTIC_ATTRIBUTES
        if column in df.columns
    ]

    for attribute in available_attributes:

        print(
            f"\n--- {MANIPULATION_COLUMN} vs {attribute} ---"
        )

        temp = df[
            [
                MANIPULATION_COLUMN,
                attribute
            ]
        ].copy()

        # Convert missing values to explicit category
        # for categorical analysis.
        temp[MANIPULATION_COLUMN] = (
            temp[MANIPULATION_COLUMN]
            .astype("object")
            .where(
                temp[MANIPULATION_COLUMN].notna(),
                "__MISSING__"
            )
        )

        temp[attribute] = (
            temp[attribute]
            .astype("object")
            .where(
                temp[attribute].notna(),
                "__MISSING__"
            )
        )

        table = pd.crosstab(
            temp[MANIPULATION_COLUMN],
            temp[attribute],
            dropna=False
        )

        print(table)

        # Calculate number of unique attribute values
        # associated with each manipulation type.
        for manipulation_type in table.index:

            row = table.loc[
                manipulation_type
            ]

            total = row.sum()

            unique_values = (
                (row > 0)
                .sum()
            )

            results.append({
                "manipulation_type":
                    manipulation_type,
                "attribute":
                    attribute,
                "unique_attribute_values":
                    int(unique_values),
                "total_records":
                    int(total)
            })

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        ATTRIBUTE_CROSSTAB_FILE,
        index=False
    )

    return result_df


# ============================================================
# 10. CHECK WHETHER MANIPULATION TYPE
#     IS A PERFECT PROXY FOR LABEL
# ============================================================

def evaluate_proxy_status(
    crosstab,
    deterministic_df
):

    print("\n")
    print("=" * 70)
    print("PROXY / DETERMINISTIC CHECK")
    print("=" * 70)

    total_categories = len(
        crosstab
    )

    deterministic_categories = len(
        deterministic_df
    )

    if total_categories > 0:

        deterministic_percentage = (
            deterministic_categories
            / total_categories
            * 100
        )

    else:

        deterministic_percentage = 0

    print(
        f"\nTotal manipulation categories: "
        f"{total_categories}"
    )

    print(
        f"Deterministic categories: "
        f"{deterministic_categories}"
    )

    print(
        f"Percentage deterministic categories: "
        f"{deterministic_percentage:.4f}%"
    )

    if deterministic_categories == total_categories:

        print(
            "\nWARNING:"
        )

        print(
            "Every manipulation_type category "
            "maps to only one label."
        )

        print(
            "This indicates a deterministic "
            "relationship between manipulation_type "
            "and label."
        )

    elif deterministic_categories > 0:

        print(
            "\nSome manipulation_type categories "
            "map to only one label."
        )

        print(
            "Further investigation is required."
        )

    else:

        print(
            "\nNo manipulation_type category "
            "is completely deterministic."
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "A deterministic relationship does not "
        "automatically prove target leakage."
    )

    print(
        "The original label-generation procedure "
        "must be inspected."
    )

    return (
        total_categories,
        deterministic_categories,
        deterministic_percentage
    )


# ============================================================
# 11. GENERATE REPORT
# ============================================================

def generate_report(
    df,
    distribution_df,
    crosstab,
    row_percentages,
    deterministic_df,
    majority_accuracy,
    majority_table,
    attribute_relationships,
    total_categories,
    deterministic_categories,
    deterministic_percentage
):

    report = []

    report.append(
        "=" * 70
    )

    report.append(
        "MANIPULATION TYPE AUDIT REPORT"
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

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    report.append(
        "\nTARGET DISTRIBUTION"
    )

    report.append(
        "-" * 70
    )

    report.append(
        df[TARGET_COLUMN]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Manipulation distribution
    # --------------------------------------------------------

    report.append(
        "\nMANIPULATION TYPE DISTRIBUTION"
    )

    report.append(
        "-" * 70
    )

    report.append(
        distribution_df
        .to_string()
    )

    # --------------------------------------------------------
    # Crosstab
    # --------------------------------------------------------

    report.append(
        "\nMANIPULATION TYPE VS LABEL - COUNTS"
    )

    report.append(
        "-" * 70
    )

    report.append(
        crosstab.to_string()
    )

    # --------------------------------------------------------
    # Row percentages
    # --------------------------------------------------------

    report.append(
        "\nMANIPULATION TYPE VS LABEL - ROW PERCENTAGES"
    )

    report.append(
        "-" * 70
    )

    report.append(
        row_percentages.to_string()
    )

    # --------------------------------------------------------
    # Deterministic categories
    # --------------------------------------------------------

    report.append(
        "\nDETERMINISTIC MANIPULATION CATEGORIES"
    )

    report.append(
        "-" * 70
    )

    if len(deterministic_df) > 0:

        report.append(
            deterministic_df.to_string(
                index=False
            )
        )

    else:

        report.append(
            "No deterministic categories found."
        )

    # --------------------------------------------------------
    # Majority analysis
    # --------------------------------------------------------

    report.append(
        "\nMAJORITY-RULE ANALYSIS"
    )

    report.append(
        "-" * 70
    )

    report.append(
        f"Majority-rule accuracy: "
        f"{majority_accuracy:.4f}%"
    )

    report.append(
        "\n"
        + majority_table.to_string()
    )

    # --------------------------------------------------------
    # Proxy status
    # --------------------------------------------------------

    report.append(
        "\nPROXY / DETERMINISTIC STATUS"
    )

    report.append(
        "-" * 70
    )

    report.append(
        f"Total categories: "
        f"{total_categories}"
    )

    report.append(
        f"Deterministic categories: "
        f"{deterministic_categories}"
    )

    report.append(
        f"Deterministic category percentage: "
        f"{deterministic_percentage:.4f}%"
    )

    # --------------------------------------------------------
    # Attribute relationships
    # --------------------------------------------------------

    report.append(
        "\nMANIPULATION TYPE VS SEMANTIC ATTRIBUTES"
    )

    report.append(
        "-" * 70
    )

    report.append(
        attribute_relationships.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    report.append(
        "\nINTERPRETATION"
    )

    report.append(
        "-" * 70
    )

    if (
        deterministic_categories
        == total_categories
        and total_categories > 0
    ):

        report.append(
            "All observed manipulation_type categories "
            "are associated with a single target label."
        )

        report.append(
            "Therefore, manipulation_type behaves as a "
            "deterministic proxy for the observed label "
            "in the current dataset."
        )

        report.append(
            "This finding requires investigation of the "
            "original label-generation procedure."
        )

    elif deterministic_categories > 0:

        report.append(
            "Some manipulation_type categories are "
            "associated with only one target label."
        )

        report.append(
            "This indicates a partial deterministic "
            "relationship and requires further investigation."
        )

    else:

        report.append(
            "No manipulation_type category is "
            "completely deterministic with respect "
            "to the target label."
        )

    report.append(
        "\nA deterministic relationship alone does "
        "not establish target leakage."
    )

    report.append(
        "Target leakage should be assessed by examining "
        "how the target label was originally constructed "
        "and whether manipulation_type represents information "
        "that would be unavailable at real-world verification time."
    )

    report.append(
        "\nIMPORTANT FOR MODELING:"
    )

    report.append(
        "manipulation_type should NOT be included as a "
        "predictor in the primary machine-learning experiment "
        "until its role in label construction has been established."
    )

    report.append(
        "\nThe script does not modify the dataset."
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    report.append(
        "\nOUTPUT FILES"
    )

    report.append(
        "-" * 70
    )

    report.append(
        f"Distribution : {DISTRIBUTION_FILE}"
    )

    report.append(
        f"Crosstab     : {CROSSTAB_FILE}"
    )

    report.append(
        f"Relationships : {ATTRIBUTE_CROSSTAB_FILE}"
    )

    report.append(
        f"Report       : {REPORT_FILE}"
    )

    return report


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Q1 SEMANTIC CREDENTIAL RESEARCH")
    print("MANIPULATION TYPE AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    check_required_columns(
        df
    )

    print_basic_information(
        df
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    distribution_df = (
        analyze_manipulation_distribution(
            df
        )
    )

    # --------------------------------------------------------
    # Manipulation vs label
    # --------------------------------------------------------

    (
        crosstab,
        row_percentages,
        deterministic_df
    ) = analyze_manipulation_vs_label(
        df
    )

    # --------------------------------------------------------
    # Majority analysis
    # --------------------------------------------------------

    (
        majority_accuracy,
        majority_table
    ) = analyze_majority_label(
        crosstab,
        df
    )

    # --------------------------------------------------------
    # Semantic attribute relationships
    # --------------------------------------------------------

    attribute_relationships = (
        analyze_attribute_relationships(
            df
        )
    )

    # --------------------------------------------------------
    # Proxy evaluation
    # --------------------------------------------------------

    (
        total_categories,
        deterministic_categories,
        deterministic_percentage
    ) = evaluate_proxy_status(
        crosstab,
        deterministic_df
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = generate_report(
        df=df,
        distribution_df=distribution_df,
        crosstab=crosstab,
        row_percentages=row_percentages,
        deterministic_df=deterministic_df,
        majority_accuracy=majority_accuracy,
        majority_table=majority_table,
        attribute_relationships=attribute_relationships,
        total_categories=total_categories,
        deterministic_categories=deterministic_categories,
        deterministic_percentage=deterministic_percentage
    )

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MANIPULATION TYPE AUDIT SELESAI")
    print("=" * 70)

    print(
        f"\nReport saved to:\n{REPORT_FILE}"
    )

    print(
        f"\nDistribution saved to:\n{DISTRIBUTION_FILE}"
    )

    print(
        f"\nCrosstab saved to:\n{CROSSTAB_FILE}"
    )

    print(
        f"\nAttribute relationships saved to:\n"
        f"{ATTRIBUTE_CROSSTAB_FILE}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Jangan masukkan manipulation_type ke model "
        "utama sebelum audit label construction selesai."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()