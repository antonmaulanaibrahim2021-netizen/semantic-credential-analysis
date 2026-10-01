from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "cross_institutional_dataset (1).xlsx"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "duplicate_test_report.txt"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("DUPLICATE AUDIT")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_excel(
    DATASET_FILE,
    sheet_name="Combined_Labeled",
    skiprows=2
)

df.columns = [str(col).strip() for col in df.columns]

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# BASIC DUPLICATE TEST
# ============================================================

exact_duplicate_count = df.duplicated().sum()

print("\n=== EXACT DUPLICATES ===")
print(f"Exact duplicate rows: {exact_duplicate_count}")


# ============================================================
# DUPLICATE BASED ON PREDICTORS
# ============================================================

TARGET = "label"

if TARGET not in df.columns:
    raise ValueError(
        "Kolom 'label' tidak ditemukan pada dataset."
    )

predictor_columns = [
    col for col in df.columns
    if col != TARGET
]

predictor_duplicate_count = df.duplicated(
    subset=predictor_columns
).sum()

print("\n=== PREDICTOR DUPLICATES ===")
print(
    "Duplicate rows based on predictors "
    f"(excluding label): {predictor_duplicate_count}"
)


# ============================================================
# SAME PREDICTORS WITH DIFFERENT LABELS
# ============================================================

print("\n=== DUPLICATE PREDICTOR WITH DIFFERENT LABEL ===")

grouped = (
    df.groupby(predictor_columns, dropna=False)[TARGET]
    .nunique()
)

conflicting_groups = grouped[grouped > 1]

print(
    "Predictor groups having both label 0 and label 1:",
    len(conflicting_groups)
)


# ============================================================
# DUPLICATE GROUP SIZE
# ============================================================

duplicate_groups = (
    df.groupby(
        predictor_columns,
        dropna=False
    )
    .size()
    .reset_index(name="count")
)

duplicate_groups = duplicate_groups[
    duplicate_groups["count"] > 1
]

duplicate_groups = duplicate_groups.sort_values(
    "count",
    ascending=False
)

print("\n=== DUPLICATE GROUP SUMMARY ===")

if len(duplicate_groups) > 0:

    print(
        f"Number of duplicated predictor groups: "
        f"{len(duplicate_groups)}"
    )

    print("\nTop duplicate groups:")
    print(
        duplicate_groups.head(20).to_string(
            index=False
        )
    )

else:

    print("No duplicated predictor groups found.")


# ============================================================
# REPORT
# ============================================================

report = []

report.append("=" * 70)
report.append("DUPLICATE AUDIT REPORT")
report.append("=" * 70)

report.append(
    f"\nDataset rows: {len(df)}"
)

report.append(
    f"Dataset columns: {len(df.columns)}"
)

report.append(
    f"\nExact duplicate rows: "
    f"{exact_duplicate_count}"
)

report.append(
    f"Predictor duplicate rows: "
    f"{predictor_duplicate_count}"
)

report.append(
    f"Predictor groups with both labels: "
    f"{len(conflicting_groups)}"
)

if len(duplicate_groups) > 0:

    report.append(
        f"\nDuplicated predictor groups: "
        f"{len(duplicate_groups)}"
    )

    report.append(
        "\nTop duplicate groups:"
    )

    report.append(
        duplicate_groups.head(20).to_string(
            index=False
        )
    )

else:

    report.append(
        "\nNo duplicated predictor groups detected."
    )


# ============================================================
# SAVE REPORT
# ============================================================

OUTPUT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("DUPLICATE AUDIT SELESAI")
print("=" * 70)

print(f"\nReport saved to:")
print(OUTPUT_FILE)

print(
    "\nJangan menghapus duplicate sebelum hasil audit "
    "ditinjau."
)