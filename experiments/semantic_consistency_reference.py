"""
SEMANTIC CONSISTENCY REFERENCE AUDIT
Q1: Semantic Consistency Assessment of Digital Academic Credentials
Using Machine Learning

Purpose
-------
Build semantic reference rules exclusively from non-manipulated
credentials (manipulation_type = "none"), then evaluate whether
manipulated credentials contain attribute combinations that are
novel relative to the normal reference set.

Reference set
-------------
manipulation_type == "none"

Test groups
-----------
1. accreditation_swap
2. education_level_swap
3. cross_institution_program

Semantic relationships examined
--------------------------------
1. institution + program
2. program + education_level
3. program + accreditation_status
4. institution + program + education_level
5. institution + program + accreditation_status

Important
---------
This is an empirical reference-based audit.

A combination not observed in the normal reference set is called
"novel", NOT automatically "invalid".

Novelty is not equivalent to semantic invalidity.

The script does NOT:
- modify the original dataset
- modify labels
- use manipulation_type as an ML predictor
- train machine-learning models
- claim that every novel combination is academically invalid

Output
------
results/semantic_reference_rules.csv
results/semantic_consistency_group_results.csv
results/semantic_consistency_record_results.csv
results/semantic_consistency_reference_report.txt
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

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

REFERENCE_RULES_FILE = (
    RESULTS_DIR
    / "semantic_reference_rules.csv"
)

GROUP_RESULTS_FILE = (
    RESULTS_DIR
    / "semantic_consistency_group_results.csv"
)

RECORD_RESULTS_FILE = (
    RESULTS_DIR
    / "semantic_consistency_record_results.csv"
)

REPORT_FILE = (
    RESULTS_DIR
    / "semantic_consistency_reference_report.txt"
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

TARGET_COLUMN = "label"

MANIPULATION_COLUMN = "manipulation_type"

NORMAL_GROUP = "none"

TEST_GROUPS = [
    "accreditation_swap",
    "education_level_swap",
    "cross_institution_program",
]


# ============================================================
# 4. SEMANTIC RELATIONSHIPS
# ============================================================

RELATIONSHIPS = {

    "institution_program": [
        "institution",
        "program",
    ],

    "program_education": [
        "program",
        "education_level",
    ],

    "program_accreditation": [
        "program",
        "accreditation_status",
    ],

    "institution_program_education": [
        "institution",
        "program",
        "education_level",
    ],

    "institution_program_accreditation": [
        "institution",
        "program",
        "accreditation_status",
    ],
}


# ============================================================
# 5. LOAD DATASET
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

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ============================================================
# 6. CHECK COLUMNS
# ============================================================

def check_columns(df):

    required = {
        TARGET_COLUMN,
        MANIPULATION_COLUMN,
    }

    for columns in RELATIONSHIPS.values():
        required.update(columns)

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
# 7. STANDARDIZE CATEGORICAL VALUES
# ============================================================

def standardize_dataset(df):

    df = df.copy()

    categorical_columns = set()

    for columns in RELATIONSHIPS.values():
        categorical_columns.update(columns)

    categorical_columns.add(
        MANIPULATION_COLUMN
    )

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
# 8. BUILD NORMAL REFERENCE SET
# ============================================================

def build_reference_set(df):

    reference_df = df[
        df[MANIPULATION_COLUMN]
        == NORMAL_GROUP
    ].copy()

    print("\n")
    print("=" * 70)
    print("REFERENCE SET")
    print("=" * 70)

    print(
        f"Normal records: {len(reference_df)}"
    )

    print(
        "\nReference group:"
    )

    print(
        reference_df[
            MANIPULATION_COLUMN
        ]
        .value_counts()
    )

    if len(reference_df) == 0:

        raise ValueError(
            "Tidak ditemukan record "
            "manipulation_type = 'none'."
        )

    return reference_df


# ============================================================
# 9. BUILD REFERENCE RULES
# ============================================================

def build_reference_rules(reference_df):

    print("\n")
    print("=" * 70)
    print("BUILDING SEMANTIC REFERENCE RULES")
    print("=" * 70)

    reference_rules = []

    for relationship_name, columns in RELATIONSHIPS.items():

        print(
            f"\nRelationship: {relationship_name}"
        )

        print(
            f"Columns: {' + '.join(columns)}"
        )

        unique_combinations = (
            reference_df[
                columns
            ]
            .drop_duplicates()
            .copy()
        )

        print(
            f"Observed normal combinations: "
            f"{len(unique_combinations)}"
        )

        for _, row in unique_combinations.iterrows():

            record = {
                "relationship":
                    relationship_name,
                "n_attributes":
                    len(columns),
                "combination":
                    " | ".join(
                        str(row[column])
                        for column in columns
                    ),
            }

            for column in columns:

                record[column] = row[column]

            reference_rules.append(
                record
            )

    rules_df = pd.DataFrame(
        reference_rules
    )

    rules_df.to_csv(
        REFERENCE_RULES_FILE,
        index=False
    )

    return rules_df


# ============================================================
# 10. CREATE REFERENCE SET LOOKUPS
# ============================================================

def create_reference_lookup(
    reference_df
):

    lookup = {}

    for relationship_name, columns in RELATIONSHIPS.items():

        combinations_set = set()

        for values in reference_df[
            columns
        ].itertuples(
            index=False,
            name=None
        ):

            combinations_set.add(
                tuple(values)
            )

        lookup[
            relationship_name
        ] = combinations_set

    return lookup


# ============================================================
# 11. TEST ONE RECORD
# ============================================================

def evaluate_record(
    row,
    reference_lookup
):

    result = {}

    for relationship_name, columns in RELATIONSHIPS.items():

        combination = tuple(
            row[column]
            for column in columns
        )

        is_observed = (
            combination
            in reference_lookup[
                relationship_name
            ]
        )

        result[
            f"{relationship_name}_observed"
        ] = int(is_observed)

        result[
            f"{relationship_name}_novel"
        ] = int(not is_observed)

    return result


# ============================================================
# 12. EVALUATE MANIPULATED RECORDS
# ============================================================

def evaluate_manipulated_records(
    df,
    reference_lookup
):

    print("\n")
    print("=" * 70)
    print("EVALUATING MANIPULATED RECORDS")
    print("=" * 70)

    manipulated_df = df[
        df[MANIPULATION_COLUMN]
        != NORMAL_GROUP
    ].copy()

    result_rows = []

    for index, row in manipulated_df.iterrows():

        evaluation = evaluate_record(
            row,
            reference_lookup
        )

        result = {
            "original_index":
                index,
            MANIPULATION_COLUMN:
                row[
                    MANIPULATION_COLUMN
                ],
            TARGET_COLUMN:
                row[
                    TARGET_COLUMN
                ],
        }

        # Include original semantic attributes.
        for relationship_columns in RELATIONSHIPS.values():

            for column in relationship_columns:

                result[column] = row[column]

        result.update(
            evaluation
        )

        # ----------------------------------------------------
        # Count novel relationships
        # ----------------------------------------------------

        novelty_columns = [
            column
            for column in evaluation
            if column.endswith(
                "_novel"
            )
        ]

        result[
            "novel_relationship_count"
        ] = sum(
            evaluation[column]
            for column in novelty_columns
        )

        result[
            "all_relationships_observed"
        ] = int(
            result[
                "novel_relationship_count"
            ] == 0
        )

        result[
            "at_least_one_novel_relationship"
        ] = int(
            result[
                "novel_relationship_count"
            ] > 0
        )

        result_rows.append(
            result
        )

    results_df = pd.DataFrame(
        result_rows
    )

    results_df.to_csv(
        RECORD_RESULTS_FILE,
        index=False
    )

    return results_df


# ============================================================
# 13. GROUP-LEVEL SUMMARY
# ============================================================

def create_group_summary(
    record_results
):

    print("\n")
    print("=" * 70)
    print("GROUP-LEVEL SEMANTIC CONSISTENCY RESULTS")
    print("=" * 70)

    rows = []

    for manipulation_type, group in (
        record_results
        .groupby(
            MANIPULATION_COLUMN
        )
    ):

        row = {
            MANIPULATION_COLUMN:
                manipulation_type,

            "records":
                len(group),

            "label_0":
                int(
                    (
                        group[TARGET_COLUMN]
                        == 0
                    ).sum()
                ),

            "label_1":
                int(
                    (
                        group[TARGET_COLUMN]
                        == 1
                    ).sum()
                ),
        }

        # ----------------------------------------------------
        # Relationship-specific novelty
        # ----------------------------------------------------

        for relationship_name in RELATIONSHIPS:

            novelty_column = (
                f"{relationship_name}_novel"
            )

            observed_column = (
                f"{relationship_name}_observed"
            )

            novelty_count = (
                group[
                    novelty_column
                ]
                .sum()
            )

            observed_count = (
                group[
                    observed_column
                ]
                .sum()
            )

            total = len(group)

            row[
                f"{relationship_name}_novel_count"
            ] = int(
                novelty_count
            )

            row[
                f"{relationship_name}_novel_percentage"
            ] = round(
                novelty_count
                / total
                * 100,
                4
            )

            row[
                f"{relationship_name}_observed_percentage"
            ] = round(
                observed_count
                / total
                * 100,
                4
            )

        # ----------------------------------------------------
        # Overall novelty
        # ----------------------------------------------------

        any_novel = (
            group[
                "at_least_one_novel_relationship"
            ]
            .sum()
        )

        all_observed = (
            group[
                "all_relationships_observed"
            ]
            .sum()
        )

        row[
            "at_least_one_novel_count"
        ] = int(
            any_novel
        )

        row[
            "at_least_one_novel_percentage"
        ] = round(
            any_novel
            / len(group)
            * 100,
            4
        )

        row[
            "all_relationships_observed_count"
        ] = int(
            all_observed
        )

        row[
            "all_relationships_observed_percentage"
        ] = round(
            all_observed
            / len(group)
            * 100,
            4
        )

        rows.append(
            row
        )

    summary_df = pd.DataFrame(
        rows
    )

    summary_df.to_csv(
        GROUP_RESULTS_FILE,
        index=False
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    return summary_df


# ============================================================
# 14. LABEL VS NOVELTY CROSS-TAB
# ============================================================

def analyze_novelty_vs_label(
    record_results
):

    print("\n")
    print("=" * 70)
    print("NOVELTY VS LABEL")
    print("=" * 70)

    cross = pd.crosstab(
        record_results[
            "at_least_one_novel_relationship"
        ],
        record_results[
            TARGET_COLUMN
        ]
    )

    print(
        "\nRows:"
    )

    print(
        "0 = all selected relationships "
        "observed in normal reference"
    )

    print(
        "1 = at least one selected relationship "
        "is novel"
    )

    print(
        "\nCross-tab:"
    )

    print(cross)

    return cross


# ============================================================
# 15. RELATIONSHIP-SPECIFIC CROSS-TABS
# ============================================================

def analyze_relationships_by_group(
    record_results
):

    print("\n")
    print("=" * 70)
    print("RELATIONSHIP-SPECIFIC ANALYSIS")
    print("=" * 70)

    results = []

    for relationship_name in RELATIONSHIPS:

        novelty_column = (
            f"{relationship_name}_novel"
        )

        print(
            f"\n--- {relationship_name} ---"
        )

        cross = pd.crosstab(
            record_results[
                MANIPULATION_COLUMN
            ],
            record_results[
                novelty_column
            ]
        )

        print(cross)

        for manipulation_type, group in (
            record_results
            .groupby(
                MANIPULATION_COLUMN
            )
        ):

            novel_count = (
                group[
                    novelty_column
                ]
                .sum()
            )

            total = len(group)

            results.append({
                "relationship":
                    relationship_name,

                "manipulation_type":
                    manipulation_type,

                "records":
                    total,

                "novel_count":
                    int(
                        novel_count
                    ),

                "novel_percentage":
                    round(
                        novel_count
                        / total
                        * 100,
                        4
                    ),
            })

    return pd.DataFrame(
        results
    )


# ============================================================
# 16. CHECK LABEL-NOVELTY CONSISTENCY
# ============================================================

def check_label_consistency(
    record_results
):

    print("\n")
    print("=" * 70)
    print("LABEL VS SEMANTIC NOVELTY CHECK")
    print("=" * 70)

    # --------------------------------------------------------
    # Label 0
    # --------------------------------------------------------

    label_zero = record_results[
        record_results[
            TARGET_COLUMN
        ] == 0
    ]

    label_zero_novel = label_zero[
        label_zero[
            "at_least_one_novel_relationship"
        ] == 1
    ]

    # --------------------------------------------------------
    # Label 1
    # --------------------------------------------------------

    label_one = record_results[
        record_results[
            TARGET_COLUMN
        ] == 1
    ]

    label_one_novel = label_one[
        label_one[
            "at_least_one_novel_relationship"
        ] == 1
    ]

    print(
        f"\nLabel 0 records: "
        f"{len(label_zero)}"
    )

    print(
        f"Label 0 with at least one novel relationship: "
        f"{len(label_zero_novel)}"
    )

    if len(label_zero) > 0:

        print(
            f"Label 0 novelty rate: "
            f"{len(label_zero_novel) / len(label_zero) * 100:.4f}%"
        )

    print(
        f"\nLabel 1 records: "
        f"{len(label_one)}"
    )

    print(
        f"Label 1 with at least one novel relationship: "
        f"{len(label_one_novel)}"
    )

    if len(label_one) > 0:

        print(
            f"Label 1 novelty rate: "
            f"{len(label_one_novel) / len(label_one) * 100:.4f}%"
        )

    return {
        "label_0_count":
            len(label_zero),

        "label_0_novel_count":
            len(label_zero_novel),

        "label_0_novel_percentage":
            (
                len(label_zero_novel)
                / len(label_zero)
                * 100
                if len(label_zero) > 0
                else 0
            ),

        "label_1_count":
            len(label_one),

        "label_1_novel_count":
            len(label_one_novel),

        "label_1_novel_percentage":
            (
                len(label_one_novel)
                / len(label_one)
                * 100
                if len(label_one) > 0
                else 0
            ),
    }


# ============================================================
# 17. IDENTIFY POTENTIALLY AMBIGUOUS RECORDS
# ============================================================

def identify_ambiguous_records(
    record_results
):

    ambiguous = record_results[
        record_results[
            "at_least_one_novel_relationship"
        ] == 0
    ].copy()

    ambiguous_file = (
        RESULTS_DIR
        / "semantic_ambiguous_records.csv"
    )

    ambiguous.to_csv(
        ambiguous_file,
        index=False
    )

    print("\n")
    print(
        "=" * 70
    )

    print(
        "AMBIGUOUS / NON-NOVEL MANIPULATED RECORDS"
    )

    print(
        "=" * 70
    )

    print(
        f"Manipulated records with no novel "
        f"relationship: {len(ambiguous)}"
    )

    print(
        f"Saved to:\n{ambiguous_file}"
    )

    return ambiguous


# ============================================================
# 18. GENERATE INTERPRETATION
# ============================================================

def generate_interpretation(
    group_summary,
    label_consistency,
    relationship_summary
):

    report = []

    report.append(
        "\nINTERPRETATION"
    )

    report.append(
        "-" * 70
    )

    report.append(
        "The reference set consists exclusively of records "
        "with manipulation_type = none."
    )

    report.append(
        "Observed attribute combinations in this reference "
        "set are treated as empirically observed normal "
        "credential patterns."
    )

    report.append(
        "A manipulated record is classified as novel for a "
        "relationship when its attribute combination does "
        "not occur in the normal reference set."
    )

    report.append(
        "\nImportant limitation:"
    )

    report.append(
        "Absence from the reference set does not prove that "
        "a credential combination is academically invalid."
    )

    report.append(
        "Novelty should therefore be interpreted as an "
        "empirical inconsistency signal rather than a "
        "formal academic validity judgment."
    )

    # --------------------------------------------------------
    # Group results
    # --------------------------------------------------------

    report.append(
        "\nGROUP-LEVEL FINDINGS"
    )

    for _, row in group_summary.iterrows():

        report.append(
            f"\n{row[MANIPULATION_COLUMN]}:"
        )

        report.append(
            f"- Records: {row['records']}"
        )

        report.append(
            "- At least one novel relationship: "
            f"{row['at_least_one_novel_percentage']:.4f}%"
        )

        report.append(
            "- All selected relationships observed: "
            f"{row['all_relationships_observed_percentage']:.4f}%"
        )

    # --------------------------------------------------------
    # Label relationship
    # --------------------------------------------------------

    report.append(
        "\nLABEL-NOVELTY FINDINGS"
    )

    report.append(
        f"Label 0 records: "
        f"{label_consistency['label_0_count']}"
    )

    report.append(
        f"Label 0 records with novelty: "
        f"{label_consistency['label_0_novel_count']}"
    )

    report.append(
        f"Label 0 novelty rate: "
        f"{label_consistency['label_0_novel_percentage']:.4f}%"
    )

    report.append(
        f"Label 1 records: "
        f"{label_consistency['label_1_count']}"
    )

    report.append(
        f"Label 1 records with novelty: "
        f"{label_consistency['label_1_novel_count']}"
    )

    report.append(
        f"Label 1 novelty rate: "
        f"{label_consistency['label_1_novel_percentage']:.4f}%"
    )

    # --------------------------------------------------------
    # Methodological conclusion
    # --------------------------------------------------------

    report.append(
        "\nMETHODOLOGICAL CONCLUSION"
    )

    report.append(
        "If label 0 records consistently show novel semantic "
        "relationships while label 1 records remain within "
        "the reference patterns, the dataset provides empirical "
        "support for interpreting the classification task as "
        "reference-based semantic consistency assessment."
    )

    report.append(
        "If substantial numbers of label 0 records remain "
        "fully observed in the normal reference set, then "
        "the label cannot be explained completely by the "
        "tested semantic relationships."
    )

    report.append(
        "Similarly, if label 1 records frequently contain "
        "novel relationships, the reference-based definition "
        "of normality requires additional investigation."
    )

    report.append(
        "\nMODEL DEVELOPMENT RESTRICTION"
    )

    report.append(
        "manipulation_type must remain excluded from the "
        "primary machine-learning feature matrix."
    )

    report.append(
        "credential_id must also remain excluded because "
        "it functions as an identifier rather than a semantic "
        "credential attribute."
    )

    return report


# ============================================================
# 19. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("Q1 SEMANTIC CREDENTIAL RESEARCH")
    print("SEMANTIC CONSISTENCY REFERENCE AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    check_columns(
        df
    )

    # --------------------------------------------------------
    # Standardize
    # --------------------------------------------------------

    df = standardize_dataset(
        df
    )

    # --------------------------------------------------------
    # Reference set
    # --------------------------------------------------------

    reference_df = (
        build_reference_set(
            df
        )
    )

    # --------------------------------------------------------
    # Reference rules
    # --------------------------------------------------------

    reference_rules = (
        build_reference_rules(
            reference_df
        )
    )

    # --------------------------------------------------------
    # Lookup
    # --------------------------------------------------------

    reference_lookup = (
        create_reference_lookup(
            reference_df
        )
    )

    # --------------------------------------------------------
    # Evaluate manipulated records
    # --------------------------------------------------------

    record_results = (
        evaluate_manipulated_records(
            df,
            reference_lookup
        )
    )

    # --------------------------------------------------------
    # Group summary
    # --------------------------------------------------------

    group_summary = (
        create_group_summary(
            record_results
        )
    )

    # --------------------------------------------------------
    # Novelty vs label
    # --------------------------------------------------------

    novelty_label_cross = (
        analyze_novelty_vs_label(
            record_results
        )
    )

    # --------------------------------------------------------
    # Relationship-specific analysis
    # --------------------------------------------------------

    relationship_summary = (
        analyze_relationships_by_group(
            record_results
        )
    )

    # --------------------------------------------------------
    # Label consistency
    # --------------------------------------------------------

    label_consistency = (
        check_label_consistency(
            record_results
        )
    )

    # --------------------------------------------------------
    # Ambiguous records
    # --------------------------------------------------------

    ambiguous_records = (
        identify_ambiguous_records(
            record_results
        )
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = []

    report.append(
        "=" * 70
    )

    report.append(
        "SEMANTIC CONSISTENCY REFERENCE AUDIT REPORT"
    )

    report.append(
        "=" * 70
    )

    report.append(
        f"\nDataset: {DATASET_FILE.name}"
    )

    report.append(
        f"Total records: {len(df)}"
    )

    report.append(
        f"Reference records: {len(reference_df)}"
    )

    report.append(
        f"Manipulated records: "
        f"{len(record_results)}"
    )

    # --------------------------------------------------------
    # Reference relationships
    # --------------------------------------------------------

    report.append(
        "\nREFERENCE RELATIONSHIPS"
    )

    report.append(
        "-" * 70
    )

    for relationship_name, columns in RELATIONSHIPS.items():

        report.append(
            f"{relationship_name}: "
            f"{len(reference_lookup[relationship_name])} "
            f"observed combinations"
        )

        report.append(
            f"Columns: {' + '.join(columns)}"
        )

    # --------------------------------------------------------
    # Group summary
    # --------------------------------------------------------

    report.append(
        "\nGROUP-LEVEL RESULTS"
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
    # Relationship summary
    # --------------------------------------------------------

    report.append(
        "\nRELATIONSHIP-SPECIFIC RESULTS"
    )

    report.append(
        "-" * 70
    )

    report.append(
        relationship_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Novelty label cross-tab
    # --------------------------------------------------------

    report.append(
        "\nNOVELTY VS LABEL"
    )

    report.append(
        "-" * 70
    )

    report.append(
        novelty_label_cross.to_string()
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    report.extend(
        generate_interpretation(
            group_summary,
            label_consistency,
            relationship_summary
        )
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
        f"Reference rules:\n"
        f"{REFERENCE_RULES_FILE}"
    )

    report.append(
        f"\nGroup results:\n"
        f"{GROUP_RESULTS_FILE}"
    )

    report.append(
        f"\nRecord results:\n"
        f"{RECORD_RESULTS_FILE}"
    )

    report.append(
        f"\nReport:\n"
        f"{REPORT_FILE}"
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
    print("SEMANTIC CONSISTENCY REFERENCE AUDIT SELESAI")
    print("=" * 70)

    print(
        f"\nReference rules:\n"
        f"{REFERENCE_RULES_FILE}"
    )

    print(
        f"\nGroup results:\n"
        f"{GROUP_RESULTS_FILE}"
    )

    print(
        f"\nRecord results:\n"
        f"{RECORD_RESULTS_FILE}"
    )

    print(
        f"\nReport:\n"
        f"{REPORT_FILE}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Novelty = kombinasi tidak ditemukan "
        "pada reference set normal."
    )

    print(
        "Novelty bukan bukti otomatis bahwa credential "
        "tidak valid secara akademik."
    )

    print(
        "manipulation_type tetap dilarang masuk "
        "ke feature matrix model utama."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()