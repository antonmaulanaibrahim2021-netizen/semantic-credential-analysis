"""
EXPERIMENT A VS B COMPARISON

Purpose:
Compare:

Experiment A:
Baseline Credential Attribute Model

Experiment B:
Semantic Knowledge Enhanced Model

Generate:
- comparison table
- improvement percentage
- JIIS manuscript table
"""


from pathlib import Path

import pandas as pd



# =====================================================
# PATH
# =====================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


RESULT_PATH = (
    PROJECT_ROOT /
    "results"
)



# =====================================================
# LOAD RESULTS
# =====================================================


def load_results():


    baseline = pd.read_csv(

        RESULT_PATH /
        "baseline_model_comparison.csv"

    )


    semantic = pd.read_csv(

        RESULT_PATH /
        "semantic_model_comparison.csv"

    )


    return baseline, semantic




# =====================================================
# COMPARISON
# =====================================================


def compare_models():



    baseline, semantic = load_results()



    metrics = [

        "Accuracy",

        "Precision",

        "Recall",

        "F1-score",

        "ROC-AUC"

    ]



    comparison = []



    for model in baseline["Model"]:



        base_row = baseline[

            baseline["Model"] == model

        ].iloc[0]



        sem_row = semantic[

            semantic["Model"] == model

        ].iloc[0]



        result = {


            "Model":

            model,


            "Baseline_F1":

            base_row["F1-score"],


            "Semantic_F1":

            sem_row["F1-score"],


            "F1_Improvement":

            (

                sem_row["F1-score"]

                -

                base_row["F1-score"]

            ),


            "Baseline_Accuracy":

            base_row["Accuracy"],


            "Semantic_Accuracy":

            sem_row["Accuracy"],


            "Accuracy_Improvement":

            (

                sem_row["Accuracy"]

                -

                base_row["Accuracy"]

            )


        }


        comparison.append(result)



    comparison_df = pd.DataFrame(

        comparison

    )



    comparison_df = comparison_df.sort_values(

        by="F1_Improvement",

        ascending=False

    )



    output = (

        RESULT_PATH /

        "experiment_A_vs_B_comparison.csv"

    )


    comparison_df.to_csv(

        output,

        index=False

    )


    print("="*70)

    print(

        "EXPERIMENT A VS B COMPARISON"

    )

    print("="*70)



    print(

        comparison_df

    )



    print(

        "\nSaved:",

        output

    )



    return comparison_df




if __name__ == "__main__":


    compare_models()