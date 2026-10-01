"""
Data Audit for Q1 Semantic Credential Research

Purpose:
1. Inspect the raw academic credential dataset before preprocessing.
2. Check structure, data types, missing values, duplicates, label distribution,
   categorical cardinality, numerical statistics, and potential target leakage.
3. Save a human-readable audit report and CSV summaries.

Run from the project root:
    python experiments/data_audit.py

Expected project structure:
    Semantic_Credential_Q1/
    ├── dataset/
    │   └── cross_institutional_dataset (1).xlsx
    ├── experiments/
    │   └── data_audit.py
    └── results/
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"

# Raw dataset used for the Q1 study
DATASET_FILE = DATASET_DIR / "cross_institutional_dataset (1).xlsx"

# Output files
REPORT_FILE = RESULTS_DIR / "data_audit_report.txt"
MISSING_FILE = RESULTS_DIR / "missing_values.csv"
DUPLICATE_FILE = RESULTS_DIR / "duplicate_summary.csv"
CATEGORY_FILE = RESULTS_DIR / "categorical_summary.csv"
NUMERIC_FILE = RESULTS_DIR / "numeric_summary.csv"
LABEL_CROSSTAB_FILE = RESULTS_DIR / "label_relationships.csv"


# ============================================================
# 2. CONFIGURATION
# ============================================================

TARGET_COLUMN = "label"

# Columns expected to be categorical based on the dataset
EXPECTED_CATEGORICAL = [
    "institution",
    "program",
    "education_level",
    "accreditation_status",
    "graduation_predicate",
    "faculty_program_name",
]

# Columns expected to be numerical
EXPECTED_NUMERICAL = [
    "gpa",
    "graduation_year",
    "graduation_month",
]


# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def section(title: str) -> str:
    line = "=" * 70
    return f"\n{line}\n{title}\n{line}\n"


def safe_pct(value, total):
    if total == 0:
        return 0.0
    return (value / total) * 100


def load_dataset():
    """Load the raw labeled Excel dataset."""
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset tidak ditemukan:\n{DATASET_FILE}\n\n"
            "Pastikan nama file dan folder sesuai."
        )

    df = pd.read_excel(
        DATASET_FILE,
        sheet_name="Combined_Labeled",
        skiprows=2
    )

    return df


def clean_column_names(df):
    """Remove accidental spaces from column names only."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


# ============================================================
# 4. MAIN AUDIT
# ============================================================

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Q1 SEMANTIC CREDENTIAL RESEARCH")
    print("DATA AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------
    print("\n[1] Loading raw dataset...")

    df = load_dataset()
    df = clean_column_names(df)

    n_rows, n_cols = df.shape

    print(f"Dataset loaded successfully.")
    print(f"Rows    : {n_rows}")
    print(f"Columns : {n_cols}")

    report = []

    report.append("=" * 70)
    report.append("Q1 SEMANTIC CREDENTIAL RESEARCH - DATA AUDIT REPORT")
    report.append("=" * 70)

    report.append(section("1. DATASET INFORMATION"))
    report.append(f"Dataset file : {DATASET_FILE.name}")
    report.append(f"Rows         : {n_rows}")
    report.append(f"Columns      : {n_cols}")
    report.append(f"Memory usage : {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    # --------------------------------------------------------
    # Column names and data types
    # --------------------------------------------------------
    print("\n[2] Checking columns and data types...")

    dtype_df = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(dtype) for dtype in df.dtypes],
        "unique_values": [df[col].nunique(dropna=False) for col in df.columns],
    })

    report.append(section("2. COLUMNS AND DATA TYPES"))
    report.append(dtype_df.to_string(index=False))

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------
    print("[3] Checking missing values...")

    missing_count = df.isna().sum()
    missing_pct = (missing_count / n_rows * 100).round(4)

    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_percentage": missing_pct.values,
    })

    missing_df.to_csv(MISSING_FILE, index=False)

    report.append(section("3. MISSING VALUES"))
    report.append(missing_df.to_string(index=False))

    # --------------------------------------------------------
    # Exact duplicates
    # --------------------------------------------------------
    print("[4] Checking exact duplicate records...")

    duplicate_mask = df.duplicated(keep=False)
    duplicate_count_all = int(duplicate_mask.sum())
    duplicate_rows = int(df.duplicated().sum())

    # Duplicate ignoring target label
    feature_columns = [c for c in df.columns if c != TARGET_COLUMN]
    duplicate_without_label = int(
        df.duplicated(subset=feature_columns).sum()
    )

    duplicate_summary = pd.DataFrame({
        "duplicate_type": [
            "Exact duplicate rows excluding first occurrence",
            "Rows belonging to duplicate groups",
            "Duplicate rows based on predictors excluding label",
        ],
        "count": [
            duplicate_rows,
            duplicate_count_all,
            duplicate_without_label,
        ],
    })

    duplicate_summary.to_csv(DUPLICATE_FILE, index=False)

    report.append(section("4. DUPLICATE ANALYSIS"))
    report.append(duplicate_summary.to_string(index=False))

    # --------------------------------------------------------
    # Target label
    # --------------------------------------------------------
    print("[5] Checking target label...")

    report.append(section("5. TARGET LABEL DISTRIBUTION"))

    if TARGET_COLUMN in df.columns:
        label_counts = df[TARGET_COLUMN].value_counts(dropna=False)
        label_percent = (
            df[TARGET_COLUMN].value_counts(dropna=False, normalize=True) * 100
        ).round(4)

        label_df = pd.DataFrame({
            "label": label_counts.index.astype(str),
            "count": label_counts.values,
            "percentage": label_percent.values,
        })

        report.append(label_df.to_string(index=False))

        print("\nLabel distribution:")
        print(label_df.to_string(index=False))
    else:
        report.append(
            f"WARNING: Target column '{TARGET_COLUMN}' was not found."
        )
        print(f"WARNING: '{TARGET_COLUMN}' not found.")

    # --------------------------------------------------------
    # Categorical variables
    # --------------------------------------------------------
    print("[6] Checking categorical variables...")

    categorical_columns = [
        col for col in EXPECTED_CATEGORICAL if col in df.columns
    ]

    category_rows = []

    report.append(section("6. CATEGORICAL VARIABLE SUMMARY"))

    for col in categorical_columns:
        n_unique = df[col].nunique(dropna=False)

        category_rows.append({
            "column": col,
            "unique_count": n_unique,
            "missing_count": int(df[col].isna().sum()),
            "top_value": (
                df[col].value_counts(dropna=False).index[0]
                if len(df[col].value_counts(dropna=False)) > 0
                else ""
            ),
            "top_count": (
                int(df[col].value_counts(dropna=False).iloc[0])
                if len(df[col].value_counts(dropna=False)) > 0
                else 0
            ),
        })

        report.append(f"\n{col}")
        report.append("-" * len(col))

        value_counts = df[col].value_counts(dropna=False)
        value_pct = (value_counts / n_rows * 100).round(2)

        category_table = pd.DataFrame({
            "value": value_counts.index.astype(str),
            "count": value_counts.values,
            "percentage": value_pct.values,
        })

        # Full values go to report, but terminal only shows first 20
        report.append(category_table.to_string(index=False))

        print(f"\n{col}: {n_unique} unique values")
        print(category_table.head(20).to_string(index=False))

    category_summary_df = pd.DataFrame(category_rows)
    category_summary_df.to_csv(CATEGORY_FILE, index=False)

    # --------------------------------------------------------
    # Numerical variables
    # --------------------------------------------------------
    print("\n[7] Checking numerical variables...")

    numerical_columns = [
        col for col in EXPECTED_NUMERICAL
        if col in df.columns
    ]

    report.append(section("7. NUMERICAL VARIABLE SUMMARY"))

    if numerical_columns:
        numeric_summary = df[numerical_columns].describe(
            include="all"
        ).T

        numeric_summary.to_csv(NUMERIC_FILE)

        report.append(numeric_summary.to_string())
        print(numeric_summary.to_string())
    else:
        pd.DataFrame().to_csv(NUMERIC_FILE)

    # --------------------------------------------------------
    # Numerical values by label
    # --------------------------------------------------------
    if TARGET_COLUMN in df.columns and numerical_columns:

        report.append(section("8. NUMERICAL VARIABLES BY LABEL"))

        grouped_numeric = (
            df.groupby(TARGET_COLUMN)[numerical_columns]
            .agg(["count", "mean", "std", "min", "max"])
        )

        report.append(grouped_numeric.to_string())

    # --------------------------------------------------------
    # Potential target leakage: categorical relationship
    # --------------------------------------------------------
    print("\n[8] Checking categorical relationships with label...")

    report.append(section(
        "9. CATEGORICAL VARIABLES VS LABEL "
        "(SCREENING FOR POSSIBLE TARGET LEAKAGE)"
    ))

    relationship_rows = []

    if TARGET_COLUMN in df.columns:

        for col in categorical_columns:

            # Cross-tab with percentages within each category.
            ctab = pd.crosstab(
                df[col],
                df[TARGET_COLUMN],
                normalize="index"
            ) * 100

            report.append(f"\n--- {col} vs {TARGET_COLUMN} ---")
            report.append(
                ctab.round(2).to_string()
            )

            # Measure maximum category-label concentration.
            max_concentration = ctab.max(axis=1).max() if not ctab.empty else np.nan

            relationship_rows.append({
                "feature": col,
                "categories": int(df[col].nunique(dropna=False)),
                "max_label_concentration_percent": round(
                    float(max_concentration), 4
                ) if not pd.isna(max_concentration) else np.nan,
            })

            # Also save detailed cross-tab as part of report only.
            # We do NOT label it as leakage automatically.

    relationship_df = pd.DataFrame(relationship_rows)
    relationship_df.to_csv(LABEL_CROSSTAB_FILE, index=False)

    # --------------------------------------------------------
    # Numerical relationship with label
    # --------------------------------------------------------
    if TARGET_COLUMN in df.columns and numerical_columns:

        report.append(section(
            "10. NUMERICAL VARIABLES VS LABEL "
            "(SCREENING)"
        ))

        for col in numerical_columns:

            report.append(f"\n--- {col} vs {TARGET_COLUMN} ---")

            try:
                grouped = df.groupby(TARGET_COLUMN)[col].agg(
                    ["count", "mean", "std", "min", "median", "max"]
                )
                report.append(grouped.to_string())
            except Exception as exc:
                report.append(f"Could not calculate: {exc}")

    # --------------------------------------------------------
    # Identifier-like columns
    # --------------------------------------------------------
    print("\n[9] Checking identifier-like columns...")

    report.append(section("11. IDENTIFIER-LIKE COLUMN SCREENING"))

    identifier_candidates = []

    for col in df.columns:
        unique_count = df[col].nunique(dropna=False)
        unique_ratio = unique_count / n_rows if n_rows else 0

        if unique_ratio >= 0.95:
            identifier_candidates.append({
                "column": col,
                "unique_count": unique_count,
                "unique_ratio": round(unique_ratio, 4),
            })

    if identifier_candidates:
        identifier_df = pd.DataFrame(identifier_candidates)
        report.append(identifier_df.to_string(index=False))
    else:
        report.append(
            "No column with >=95% unique values was detected."
        )

    # --------------------------------------------------------
    # Constant / near-constant columns
    # --------------------------------------------------------
    print("[10] Checking constant and near-constant columns...")

    report.append(section(
        "12. CONSTANT AND NEAR-CONSTANT VARIABLES"
    ))

    constant_rows = []

    for col in df.columns:
        counts = df[col].value_counts(dropna=False)

        if len(counts) == 0:
            continue

        top_pct = counts.iloc[0] / n_rows * 100

        if len(counts) == 1 or top_pct >= 99:
            constant_rows.append({
                "column": col,
                "unique_count": int(df[col].nunique(dropna=False)),
                "dominant_value_percentage": round(top_pct, 4),
            })

    if constant_rows:
        report.append(
            pd.DataFrame(constant_rows).to_string(index=False)
        )
    else:
        report.append(
            "No constant or >=99% dominant variable detected."
        )

    # --------------------------------------------------------
    # Data consistency checks
    # --------------------------------------------------------
    print("[11] Checking basic value consistency...")

    report.append(section("13. BASIC VALUE CONSISTENCY CHECKS"))

    if "gpa" in df.columns:
        invalid_gpa = df[
            df["gpa"].notna() &
            ((df["gpa"] < 0) | (df["gpa"] > 4))
        ]

        report.append(
            f"GPA outside 0-4 range: {len(invalid_gpa)} rows"
        )

    if "graduation_month" in df.columns:
        invalid_month = df[
            df["graduation_month"].notna() &
            (
                (df["graduation_month"] < 1) |
                (df["graduation_month"] > 12)
            )
        ]

        report.append(
            f"Graduation month outside 1-12 range: "
            f"{len(invalid_month)} rows"
        )

    if "graduation_year" in df.columns:
        invalid_year = df[
            df["graduation_year"].notna() &
            (
                (df["graduation_year"] < 1900) |
                (df["graduation_year"] > 2100)
            )
        ]

        report.append(
            f"Graduation year outside 1900-2100 range: "
            f"{len(invalid_year)} rows"
        )

    # --------------------------------------------------------
    # Example rows
    # --------------------------------------------------------
    report.append(section("14. SAMPLE RECORDS"))
    report.append(df.head(10).to_string(index=True))

    # --------------------------------------------------------
    # Preliminary interpretation
    # --------------------------------------------------------
    report.append(section("15. PRELIMINARY AUDIT INTERPRETATION"))

    report.append(
        "This section is a screening result, not a final leakage diagnosis."
    )

    report.append(
        f"- Total records: {n_rows}"
    )

    report.append(
        f"- Total variables: {n_cols}"
    )

    report.append(
        f"- Exact duplicate rows excluding first occurrence: "
        f"{duplicate_rows}"
    )

    report.append(
        f"- Duplicate rows based on predictors excluding label: "
        f"{duplicate_without_label}"
    )

    if TARGET_COLUMN in df.columns:
        report.append(
            f"- Target column '{TARGET_COLUMN}' exists."
        )
    else:
        report.append(
            f"- WARNING: target column '{TARGET_COLUMN}' does not exist."
        )

    if len(constant_rows) > 0:
        report.append(
            "- At least one constant/near-constant variable was detected. "
            "Review it before modeling."
        )
    else:
        report.append(
            "- No constant/near-constant variable was detected under the "
            "screening threshold."
        )

    if duplicate_without_label > 0:
        report.append(
            "- Predictor duplicates exist. These require special attention "
            "when creating train/test splits because identical predictor "
            "patterns can cross partitions."
        )

    report.append(
        "\nIMPORTANT: A strong relationship between a feature and the label "
        "does not by itself prove target leakage. Leakage must be evaluated "
        "from the data-generation process and the modeling pipeline."
    )

    report.append(
        "\nNext step: use this report to design a separate leakage test before "
        "running the final Q1 model comparison."
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------
    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("DATA AUDIT SELESAI")
    print("=" * 70)

    print(f"\nReport : {REPORT_FILE}")
    print(f"Missing: {MISSING_FILE}")
    print(f"Duplicate: {DUPLICATE_FILE}")
    print(f"Categorical: {CATEGORY_FILE}")
    print(f"Numerical: {NUMERIC_FILE}")
    print(f"Label relationships: {LABEL_CROSSTAB_FILE}")

    print("\nJangan melakukan preprocessing/modeling dari script ini.")
    print("Gunakan hasil audit sebagai dasar untuk leakage_test.py.")


if __name__ == "__main__":
    main()
