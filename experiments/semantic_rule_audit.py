"""
SEMANTIC RULE AUDIT
Q1: Semantic Consistency Assessment of Digital Academic Credentials
Using Machine Learning

Purpose
-------
This script investigates whether the synthetic manipulation types
correspond to observable semantic inconsistencies in academic
credential attributes.

The audit examines:

1. accreditation_swap
2. education_level_swap
3. cross_institution_program
4. none

It compares each manipulation group with the non-manipulated
credential group and evaluates:

- categorical distributions
- institution-program consistency
- institution-faculty/program consistency
- program-education-level relationships
- program-accreditation relationships
- education-level-accreditation relationships
- missing-value patterns
- numerical GPA distributions
- graduation-date patterns

IMPORTANT
---------
This script is descriptive.

It does NOT:
- modify the dataset
- remove records
- modify labels
- train ML models
- use manipulation_type as an ML predictor
- automatically declare target leakage

The purpose is to determine whether the observed manipulation
types have a meaningful semantic interpretation.
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
    / "semantic_rule_audit_report.txt"
)

GROUP_SUMMARY_FILE = (
    RESULTS_DIR
    / "semantic_rule_group_summary.csv"
)

PROGRAM_INSTITUTION_FILE = (
    RESULTS_DIR
    / "program_institution_consistency.csv"
)

PROGRAM_EDUCATION_FILE = (
    RESULTS_DIR
    / "program_education_relationship.csv"
)

PROGRAM_ACCREDITATION_FILE = (
    RESULTS_DIR
    / "program_accreditation_relationship.csv"
)

EDUCATION_ACCREDITATION_FILE = (
    RESULTS_DIR
    / "education_accreditation_relationship.csv"
)

ATTRIBUTE_SUMMARY_FILE = (
    RESULTS_DIR
    / "semantic_attribute_summary.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TARGET_COLUMN = "label"

MANIPULATION_COLUMN = "manipulation_type"

INSTITUTION = "institution"
PROGRAM = "program"
EDUCATION = "education_level"
ACCREDITATION = "accreditation_status"
GPA = "gpa"
GRADUATION_PREDICATE = "graduation_predicate"
GRADUATION_DATE = "graduation_date"
FACULTY_PROGRAM = "faculty_program_name"


MANIPULATION_TYPES = [
    "none",
    "accreditation_swap",
    "education_level_swap",
    "cross_institution_program",
]


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset():

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            "\nDataset tidak ditemukan:\n"
            f"{DATASET_FILE}"
        )

    print("\nLoading dataset...")

    df = pd.read_excel(
        DATASET_FILE,
        sheet_name="Combined_Labeled",
        skiprows=2
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

def check_columns(df):

    required = [
        TARGET_COLUMN,
        MANIPULATION_COLUMN,
        INSTITUTION,
        PROGRAM,
        EDUCATION,
        ACCREDITATION,
        GPA,
        GRADUATION_PREDICATE,
        GRADUATION_DATE,
        FACULTY_PROGRAM,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nKolom berikut tidak ditemukan:\n"
            + "\n".join(
                f"- {column}"
                for column in missing
            )
        )


# ============================================================
# 5. STANDARDIZE VALUES FOR AUDIT
# ============================================================

def standardize_values(df):

    df = df.copy()

    categorical_columns = [
        INSTITUTION,
        PROGRAM,
        EDUCATION,
        ACCREDITATION,
        GRADUATION_PREDICATE,
        FACULTY_PROGRAM,
        MANIPULATION_COLUMN,
    ]

    for column in categorical_columns:

        df[column] = (
            df[column]
            .astype("object")
            .where(
                df[column].notna(),
                "__MISSING__"
            )
        )

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    return df


# ============================================================
# 6. BASIC GROUP SUMMARY
# ============================================================

def create_group_summary(df):

    print("\n")
    print("=" * 70)
    print("MANIPULATION GROUP SUMMARY")
    print("=" * 70)

    rows = []

    for manipulation_type, group in df.groupby(
        MANIPULATION_COLUMN,
        dropna=False
    ):

        row = {
            "manipulation_type":
                manipulation_type,
            "count":
                len(group),
            "percentage":
                round(
                    len(group) / len(df) * 100,
                    4
                ),
            "label_0":
                int(
                    (group[TARGET_COLUMN] == 0)
                    .sum()
                ),
            "label_1":
                int(
                    (group[TARGET_COLUMN] == 1)
                    .sum()
                ),
        }

        rows.append(row)

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False
        )
    )

    result.to_csv(
        GROUP_SUMMARY_FILE,
        index=False
    )

    return result


# ============================================================
# 7. MISSING VALUE ANALYSIS
# ============================================================

def analyze_missing_values(df):

    print("\n")
    print("=" * 70)
    print("MISSING VALUE ANALYSIS BY MANIPULATION TYPE")
    print("=" * 70)

    attributes = [
        INSTITUTION,
        PROGRAM,
        EDUCATION,
        ACCREDITATION,
        GPA,
        GRADUATION_PREDICATE,
        GRADUATION_DATE,
        FACULTY_PROGRAM,
    ]

    rows = []

    for manipulation_type, group in df.groupby(
        MANIPULATION_COLUMN,
        dropna=False
    ):

        for attribute in attributes:

            missing_count = (
                group[attribute]
                .isna()
                .sum()
            )

            missing_percentage = (
                missing_count
                / len(group)
                * 100
                if len(group) > 0
                else 0
            )

            rows.append({
                "manipulation_type":
                    manipulation_type,
                "attribute":
                    attribute,
                "missing_count":
                    int(missing_count),
                "missing_percentage":
                    round(
                        missing_percentage,
                        4
                    )
            })

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# 8. PROGRAM-INSTITUTION CONSISTENCY
# ============================================================

def analyze_program_institution(df):

    print("\n")
    print("=" * 70)
    print("PROGRAM-INSTITUTION RELATIONSHIP")
    print("=" * 70)

    result = (
        pd.crosstab(
            df[PROGRAM],
            df[INSTITUTION]
        )
    )

    print("\nProgram by institution:")
    print(result)

    # --------------------------------------------------------
    # Count institutions per program
    # --------------------------------------------------------

    program_institution_count = (
        df.groupby(
            PROGRAM
        )[INSTITUTION]
        .nunique()
        .reset_index(
            name="institution_count"
        )
    )

    program_institution_count[
        "cross_institution_program_candidate"
    ] = (
        program_institution_count[
            "institution_count"
        ] > 1
    )

    print(
        "\nPrograms associated with multiple institutions:"
    )

    print(
        program_institution_count[
            program_institution_count[
                "institution_count"
            ] > 1
        ]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        PROGRAM_INSTITUTION_FILE
    )

    return (
        result,
        program_institution_count
    )


# ============================================================
# 9. PROGRAM-EDUCATION RELATIONSHIP
# ============================================================

def analyze_program_education(df):

    print("\n")
    print("=" * 70)
    print("PROGRAM-EDUCATION LEVEL RELATIONSHIP")
    print("=" * 70)

    result = pd.crosstab(
        df[PROGRAM],
        df[EDUCATION]
    )

    print(result)

    # Number of education levels per program.
    program_education_count = (
        df.groupby(
            PROGRAM
        )[EDUCATION]
        .nunique()
        .reset_index(
            name="education_level_count"
        )
    )

    print(
        "\nPrograms associated with multiple "
        "education levels:"
    )

    print(
        program_education_count[
            program_education_count[
                "education_level_count"
            ] > 1
        ]
        .to_string(index=False)
    )

    result.to_csv(
        PROGRAM_EDUCATION_FILE
    )

    return (
        result,
        program_education_count
    )


# ============================================================
# 10. PROGRAM-ACCREDITATION RELATIONSHIP
# ============================================================

def analyze_program_accreditation(df):

    print("\n")
    print("=" * 70)
    print("PROGRAM-ACCREDITATION RELATIONSHIP")
    print("=" * 70)

    result = pd.crosstab(
        df[PROGRAM],
        df[ACCREDITATION]
    )

    print(result)

    program_accreditation_count = (
        df.groupby(
            PROGRAM
        )[ACCREDITATION]
        .nunique()
        .reset_index(
            name="accreditation_status_count"
        )
    )

    print(
        "\nPrograms associated with multiple "
        "accreditation statuses:"
    )

    print(
        program_accreditation_count[
            program_accreditation_count[
                "accreditation_status_count"
            ] > 1
        ]
        .to_string(index=False)
    )

    result.to_csv(
        PROGRAM_ACCREDITATION_FILE
    )

    return (
        result,
        program_accreditation_count
    )


# ============================================================
# 11. EDUCATION-ACCREDITATION RELATIONSHIP
# ============================================================

def analyze_education_accreditation(df):

    print("\n")
    print("=" * 70)
    print("EDUCATION-ACCREDITATION RELATIONSHIP")
    print("=" * 70)

    result = pd.crosstab(
        df[EDUCATION],
        df[ACCREDITATION]
    )

    print(result)

    result.to_csv(
        EDUCATION_ACCREDITATION_FILE
    )

    return result


# ============================================================
# 12. MANIPULATION-SPECIFIC ATTRIBUTE PROFILE
# ============================================================

def create_attribute_profile(df):

    print("\n")
    print("=" * 70)
    print("MANIPULATION-SPECIFIC ATTRIBUTE PROFILE")
    print("=" * 70)

    rows = []

    for manipulation_type, group in df.groupby(
        MANIPULATION_COLUMN,
        dropna=False
    ):

        row = {
            "manipulation_type":
                manipulation_type,
            "count":
                len(group),
        }

        # ----------------------------------------------------
        # Categorical diversity
        # ----------------------------------------------------

        for attribute in [
            INSTITUTION,
            PROGRAM,
            EDUCATION,
            ACCREDITATION,
            GRADUATION_PREDICATE,
            FACULTY_PROGRAM,
        ]:

            row[
                f"{attribute}_unique"
            ] = (
                group[attribute]
                .nunique(dropna=False)
            )

        # ----------------------------------------------------
        # GPA
        # ----------------------------------------------------

        row["gpa_count"] = (
            group[GPA]
            .notna()
            .sum()
        )

        row["gpa_missing"] = (
            group[GPA]
            .isna()
            .sum()
        )

        row["gpa_mean"] = (
            group[GPA]
            .mean()
        )

        row["gpa_std"] = (
            group[GPA]
            .std()
        )

        row["gpa_min"] = (
            group[GPA]
            .min()
        )

        row["gpa_max"] = (
            group[GPA]
            .max()
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# 13. CHECK SPECIFIC MANIPULATION PATTERNS
# ============================================================

def inspect_manipulation_patterns(df):

    print("\n")
    print("=" * 70)
    print("MANIPULATION-SPECIFIC PATTERN INSPECTION")
    print("=" * 70)

    results = {}

    # --------------------------------------------------------
    # Accreditation swap
    # --------------------------------------------------------

    accreditation_group = df[
        df[MANIPULATION_COLUMN]
        == "accreditation_swap"
    ]

    print("\n[1] ACCREDITATION SWAP")

    if len(accreditation_group) > 0:

        print(
            "Accreditation distribution:"
        )

        print(
            accreditation_group[
                ACCREDITATION
            ]
            .value_counts(
                dropna=False
            )
        )

        print(
            "\nProgram distribution:"
        )

        print(
            accreditation_group[
                PROGRAM
            ]
            .value_counts()
            .head(20)
        )

    else:

        print(
            "No accreditation_swap records found."
        )

    # --------------------------------------------------------
    # Education level swap
    # --------------------------------------------------------

    education_group = df[
        df[MANIPULATION_COLUMN]
        == "education_level_swap"
    ]

    print("\n[2] EDUCATION LEVEL SWAP")

    if len(education_group) > 0:

        print(
            "Education-level distribution:"
        )

        print(
            education_group[
                EDUCATION
            ]
            .value_counts(
                dropna=False
            )
        )

        print(
            "\nProgram distribution:"
        )

        print(
            education_group[
                PROGRAM
            ]
            .value_counts()
            .head(20)
        )

    else:

        print(
            "No education_level_swap records found."
        )

    # --------------------------------------------------------
    # Cross-institution program
    # --------------------------------------------------------

    cross_group = df[
        df[MANIPULATION_COLUMN]
        == "cross_institution_program"
    ]

    print(
        "\n[3] CROSS-INSTITUTION PROGRAM"
    )

    if len(cross_group) > 0:

        print(
            "Institution distribution:"
        )

        print(
            cross_group[
                INSTITUTION
            ]
            .value_counts(
                dropna=False
            )
        )

        print(
            "\nProgram distribution:"
        )

        print(
            cross_group[
                PROGRAM
            ]
            .value_counts()
            .head(20)
        )

        print(
            "\nFaculty/program distribution:"
        )

        print(
            cross_group[
                FACULTY_PROGRAM
            ]
            .value_counts(
                dropna=False
            )
            .head(20)
        )

    else:

        print(
            "No cross_institution_program "
            "records found."
        )

    results[
        "accreditation_swap_count"
    ] = len(accreditation_group)

    results[
        "education_level_swap_count"
    ] = len(education_group)

    results[
        "cross_institution_program_count"
    ] = len(cross_group)

    return results


# ============================================================
# 14. COMPARE MANIPULATED VS NON-MANIPULATED
# ============================================================

def compare_manipulated_vs_normal(df):

    print("\n")
    print("=" * 70)
    print("MANIPULATED VS NON-MANIPULATED PROFILE")
    print("=" * 70)

    normal = df[
        df[MANIPULATION_COLUMN]
        == "none"
    ]

    manipulated = df[
        df[MANIPULATION_COLUMN]
        != "none"
    ]

    rows = []

    categorical_attributes = [
        INSTITUTION,
        PROGRAM,
        EDUCATION,
        ACCREDITATION,
        GRADUATION_PREDICATE,
        FACULTY_PROGRAM,
    ]

    for attribute in categorical_attributes:

        normal_distribution = (
            normal[attribute]
            .value_counts(
                normalize=True,
                dropna=False
            )
            * 100
        )

        manipulated_distribution = (
            manipulated[attribute]
            .value_counts(
                normalize=True,
                dropna=False
            )
            * 100
        )

        all_values = sorted(
            set(
                normal_distribution.index
            )
            |
            set(
                manipulated_distribution.index
            ),
            key=str
        )

        for value in all_values:

            rows.append({
                "attribute":
                    attribute,
                "value":
                    value,
                "normal_percentage":
                    round(
                        normal_distribution.get(
                            value,
                            0
                        ),
                        4
                    ),
                "manipulated_percentage":
                    round(
                        manipulated_distribution.get(
                            value,
                            0
                        ),
                        4
                    ),
            })

    result = pd.DataFrame(rows)

    return result


# ============================================================
# 15. GENERATE INTERPRETATION
# ============================================================

def generate_interpretation(
    df,
    program_institution_count,
    program_education_count,
    program_accreditation_count,
    attribute_profile
):

    report = []

    report.append(
        "\nINTERPRETATION"
    )

    report.append(
        "-" * 70
    )

    report.append(
        "The semantic rule audit evaluates whether the observed "
        "manipulation categories correspond to observable changes "
        "or inconsistencies in academic credential attributes."
    )

    report.append(
        "\n1. Accreditation swap"
    )

    accreditation_group = df[
        df[MANIPULATION_COLUMN]
        == "accreditation_swap"
    ]

    if len(accreditation_group) > 0:

        report.append(
            f"- Records analyzed: "
            f"{len(accreditation_group)}"
        )

        report.append(
            "- The accreditation distribution was examined "
            "against the non-manipulated group."
        )

    report.append(
        "\n2. Education-level swap"
    )

    education_group = df[
        df[MANIPULATION_COLUMN]
        == "education_level_swap"
    ]

    if len(education_group) > 0:

        report.append(
            f"- Records analyzed: "
            f"{len(education_group)}"
        )

        report.append(
            "- Education-level distributions were examined "
            "against the non-manipulated group."
        )

    report.append(
        "\n3. Cross-institution program"
    )

    cross_group = df[
        df[MANIPULATION_COLUMN]
        == "cross_institution_program"
    ]

    if len(cross_group) > 0:

        report.append(
            f"- Records analyzed: "
            f"{len(cross_group)}"
        )

        multi_institution_programs = (
            program_institution_count[
                program_institution_count[
                    "institution_count"
                ] > 1
            ]
        )

        report.append(
            f"- Programs associated with more than "
            f"one institution: "
            f"{len(multi_institution_programs)}"
        )

    report.append(
        "\n4. Program-education relationship"
    )

    report.append(
        f"- Programs associated with multiple "
        f"education levels: "
        f"{len(program_education_count[
            program_education_count[
                'education_level_count'
            ] > 1
        ])}"
    )

    report.append(
        "\n5. Program-accreditation relationship"
    )

    report.append(
        f"- Programs associated with multiple "
        f"accreditation statuses: "
        f"{len(program_accreditation_count[
            program_accreditation_count[
                'accreditation_status_count'
            ] > 1
        ])}"
    )

    report.append(
        "\n6. Important methodological interpretation"
    )

    report.append(
        "The presence of manipulated records does not by "
        "itself establish that the observed machine-learning "
        "task represents real-world semantic validation."
    )

    report.append(
        "The audit is intended to establish whether the "
        "synthetic manipulation procedures correspond to "
        "observable semantic inconsistencies."
    )

    report.append(
        "If the manipulation procedures directly define the "
        "target label, the resulting classification task should "
        "be described as synthetic manipulation-based semantic "
        "consistency classification rather than unrestricted "
        "real-world credential verification."
    )

    report.append(
        "The variable manipulation_type must remain excluded "
        "from the primary ML feature set because it is a "
        "deterministic proxy for the target label."
    )

    return report


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Q1 SEMANTIC CREDENTIAL RESEARCH")
    print("SEMANTIC RULE AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    check_columns(df)

    print(
        f"\nRows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Standardize
    # --------------------------------------------------------

    df = standardize_values(
        df
    )

    # --------------------------------------------------------
    # Group summary
    # --------------------------------------------------------

    group_summary = (
        create_group_summary(
            df
        )
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_summary = (
        analyze_missing_values(
            df
        )
    )

    # --------------------------------------------------------
    # Program-institution
    # --------------------------------------------------------

    (
        program_institution_table,
        program_institution_count
    ) = analyze_program_institution(
        df
    )

    # --------------------------------------------------------
    # Program-education
    # --------------------------------------------------------

    (
        program_education_table,
        program_education_count
    ) = analyze_program_education(
        df
    )

    # --------------------------------------------------------
    # Program-accreditation
    # --------------------------------------------------------

    (
        program_accreditation_table,
        program_accreditation_count
    ) = analyze_program_accreditation(
        df
    )

    # --------------------------------------------------------
    # Education-accreditation
    # --------------------------------------------------------

    education_accreditation_table = (
        analyze_education_accreditation(
            df
        )
    )

    # --------------------------------------------------------
    # Attribute profile
    # --------------------------------------------------------

    attribute_profile = (
        create_attribute_profile(
            df
        )
    )

    attribute_profile.to_csv(
        ATTRIBUTE_SUMMARY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Manipulation-specific inspection
    # --------------------------------------------------------

    manipulation_patterns = (
        inspect_manipulation_patterns(
            df
        )
    )

    # --------------------------------------------------------
    # Normal vs manipulated
    # --------------------------------------------------------

    comparison = (
        compare_manipulated_vs_normal(
            df
        )
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = []

    report.append(
        "=" * 70
    )

    report.append(
        "SEMANTIC RULE AUDIT REPORT"
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
        "\nDataset columns:"
    )

    for column in df.columns:

        report.append(
            f"- {column}"
        )

    # --------------------------------------------------------
    # Group summary
    # --------------------------------------------------------

    report.append(
        "\nMANIPULATION GROUP SUMMARY"
    )

    report.append(
        "-" * 70
    )

    report.append(
        group_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Missing summary
    # --------------------------------------------------------

    report.append(
        "\nMISSING VALUE SUMMARY"
    )

    report.append(
        "-" * 70
    )

    report.append(
        missing_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Program-institution
    # --------------------------------------------------------

    report.append(
        "\nPROGRAM-INSTITUTION RELATIONSHIP"
    )

    report.append(
        "-" * 70
    )

    report.append(
        program_institution_table.to_string()
    )

    # --------------------------------------------------------
    # Program-education
    # --------------------------------------------------------

    report.append(
        "\nPROGRAM-EDUCATION RELATIONSHIP"
    )

    report.append(
        "-" * 70
    )

    report.append(
        program_education_table.to_string()
    )

    # --------------------------------------------------------
    # Program-accreditation
    # --------------------------------------------------------

    report.append(
        "\nPROGRAM-ACCREDITATION RELATIONSHIP"
    )

    report.append(
        "-" * 70
    )

    report.append(
        program_accreditation_table.to_string()
    )

    # --------------------------------------------------------
    # Education-accreditation
    # --------------------------------------------------------

    report.append(
        "\nEDUCATION-ACCREDITATION RELATIONSHIP"
    )

    report.append(
        "-" * 70
    )

    report.append(
        education_accreditation_table.to_string()
    )

    # --------------------------------------------------------
    # Attribute profile
    # --------------------------------------------------------

    report.append(
        "\nATTRIBUTE PROFILE BY MANIPULATION TYPE"
    )

    report.append(
        "-" * 70
    )

    report.append(
        attribute_profile.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    report.append(
        "\nMANIPULATED VS NON-MANIPULATED"
    )

    report.append(
        "-" * 70
    )

    report.append(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Manipulation pattern summary
    # --------------------------------------------------------

    report.append(
        "\nMANIPULATION PATTERN COUNTS"
    )

    report.append(
        "-" * 70
    )

    for key, value in manipulation_patterns.items():

        report.append(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    report.extend(
        generate_interpretation(
            df,
            program_institution_count,
            program_education_count,
            program_accreditation_count,
            attribute_profile
        )
    )

    # --------------------------------------------------------
    # Important modeling restriction
    # --------------------------------------------------------

    report.append(
        "\nMODELING RESTRICTIONS"
    )

    report.append(
        "-" * 70
    )

    report.append(
        "1. credential_id must not be used as a predictor."
    )

    report.append(
        "2. manipulation_type must not be used as a predictor."
    )

    report.append(
        "3. label must remain the target variable."
    )

    report.append(
        "4. Preprocessing must be performed after the "
        "train-test split to avoid data leakage."
    )

    report.append(
        "5. Any semantic rule inferred from this audit "
        "must be documented before final model evaluation."
    )

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    report.append(
        "\nOUTPUT FILES"
    )

    report.append(
        "-" * 70
    )

    report.append(
        f"Group summary:"
        f"\n{GROUP_SUMMARY_FILE}"
    )

    report.append(
        f"\nProgram-institution:"
        f"\n{PROGRAM_INSTITUTION_FILE}"
    )

    report.append(
        f"\nProgram-education:"
        f"\n{PROGRAM_EDUCATION_FILE}"
    )

    report.append(
        f"\nProgram-accreditation:"
        f"\n{PROGRAM_ACCREDITATION_FILE}"
    )

    report.append(
        f"\nEducation-accreditation:"
        f"\n{EDUCATION_ACCREDITATION_FILE}"
    )

    report.append(
        f"\nAttribute summary:"
        f"\n{ATTRIBUTE_SUMMARY_FILE}"
    )

    report.append(
        f"\nFull report:"
        f"\n{REPORT_FILE}"
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SEMANTIC RULE AUDIT SELESAI")
    print("=" * 70)

    print(
        f"\nReport saved to:\n"
        f"{REPORT_FILE}"
    )

    print(
        "\nAudit selesai tanpa mengubah dataset."
    )

    print(
        "\nJangan masukkan manipulation_type "
        "ke model utama."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()