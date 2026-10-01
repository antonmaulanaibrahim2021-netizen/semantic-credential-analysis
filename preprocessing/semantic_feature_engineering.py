"""
SEMANTIC FEATURE ENGINEERING

Project:
Semantic_Credential_Q1


Experiment B:
Semantic Knowledge Enhanced Model


Purpose:
Combine original credential attributes
with semantic consistency features.
"""


from pathlib import Path

import pandas as pd
import numpy as np



# ==========================================================
# PATH
# ==========================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATASET_PATH = (
    PROJECT_ROOT
    /
    "dataset"
    /
    "cross_institutional_dataset (1).xlsx"
)


RESULT_PATH = (
    PROJECT_ROOT
    /
    "results"
)



# ==========================================================
# LOAD ORIGINAL DATA
# ==========================================================


def load_dataset():


    df = pd.read_excel(

        DATASET_PATH,

        sheet_name="Combined_Labeled",

        header=2

    )


    df.columns = (

        df.columns

        .astype(str)

        .str.strip()

        .str.lower()

    )


    return df




# ==========================================================
# LOAD SEMANTIC RESULT
# ==========================================================


def load_semantic_results():


    files = {


        "reference":

        "semantic_reference_rules.csv",


        "record":

        "semantic_consistency_record_results.csv",


        "group":

        "semantic_consistency_group_results.csv"

    }


    semantic = {}


    for key,file in files.items():


        path = RESULT_PATH / file


        if path.exists():

            semantic[key] = pd.read_csv(path)


            print(
                f"Loaded {file}"
            )

        else:

            print(
                f"Missing: {file}"
            )


    return semantic




# ==========================================================
# CREATE SEMANTIC FEATURES
# ==========================================================


def create_semantic_features(df):


    df = df.copy()



    # ----------------------------------
    # 1 Institution Program
    # ----------------------------------


    df["institution_program_key"] = (

        df["institution"].astype(str)

        + "_"

        + df["program"].astype(str)

    )



    normal_ip = (

        df[df["manipulation_type"]=="none"]

        ["institution_program_key"]

        .unique()

    )


    df["institution_program_valid"] = (

        df["institution_program_key"]

        .isin(normal_ip)

        .astype(int)

    )



    # ----------------------------------
    # 2 Program Education
    # ----------------------------------


    df["program_education_key"] = (

        df["program"].astype(str)

        + "_"

        + df["education_level"].astype(str)

    )



    normal_pe = (

        df[df["manipulation_type"]=="none"]

        ["program_education_key"]

        .unique()

    )


    df["program_education_valid"] = (

        df["program_education_key"]

        .isin(normal_pe)

        .astype(int)

    )



    # ----------------------------------
    # 3 Program Accreditation
    # ----------------------------------


    df["program_accreditation_key"] = (

        df["program"].astype(str)

        + "_"

        + df["accreditation_status"].astype(str)

    )


    normal_pa = (

        df[df["manipulation_type"]=="none"]

        ["program_accreditation_key"]

        .unique()

    )


    df["program_accreditation_valid"] = (

        df["program_accreditation_key"]

        .isin(normal_pa)

        .astype(int)

    )



    # ----------------------------------
    # Semantic Score
    # ----------------------------------


    semantic_columns = [

        "institution_program_valid",

        "program_education_valid",

        "program_accreditation_valid"

    ]


    df["semantic_consistency_score"] = (

        df[semantic_columns]

        .sum(axis=1)

        /
        len(semantic_columns)

    )



    return df




# ==========================================================
# PREPARE SEMANTIC DATASET
# ==========================================================


def prepare_semantic_dataset():


    print("="*70)

    print(
        "SEMANTIC FEATURE ENGINEERING"
    )

    print("="*70)



    df = load_dataset()



    print(
        "Original:",
        df.shape
    )



    df = create_semantic_features(df)



    # simpan hasil


    output = (

        RESULT_PATH

        /
        "semantic_feature_dataset.csv"

    )


    df.to_csv(

        output,

        index=False

    )


    print(

        "Saved:",

        output

    )


    print(

        "Final shape:",

        df.shape

    )


    return df




# ==========================================================
# TEST
# ==========================================================


if __name__ == "__main__":


    df = prepare_semantic_dataset()


    print(

        df[

        [

        "institution_program_valid",

        "program_education_valid",

        "program_accreditation_valid",

        "semantic_consistency_score"

        ]

        ].head()

    )