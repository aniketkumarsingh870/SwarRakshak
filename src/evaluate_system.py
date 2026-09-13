import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from feature_extractor import (
    extract_features,
    get_feature_names
)


# ==========================================
# PATHS
# ==========================================

REAL_EVAL_FOLDER = "data/evaluation/real"
FAKE_EVAL_FOLDER = "data/evaluation/fake"

MODEL_PATH = "models/swarrakshak_model.pkl"
SCALER_PATH = "models/swarrakshak_scaler.pkl"

OUTPUT_CSV = "data/evaluation/evaluation_results.csv"


# ==========================================
# RISK LEVEL FUNCTION
# ==========================================

def get_risk_level(score):

    if score < 30:
        return "LOW"

    elif score < 55:
        return "MEDIUM"

    elif score < 80:
        return "HIGH"

    else:
        return "CRITICAL"


# ==========================================
# PROCESS ONE FOLDER
# ==========================================

def process_folder(
    folder,
    true_label,
    model,
    scaler,
    feature_names
):

    results = []

    if not os.path.exists(folder):

        print(
            f"\nFolder not found: {folder}"
        )

        return results

    files = [
        file
        for file in os.listdir(folder)
        if file.lower().endswith(".wav")
    ]

    print(
        f"\nFiles found in {folder}: {len(files)}"
    )

    for index, file_name in enumerate(
        files,
        start=1
    ):

        file_path = os.path.join(
            folder,
            file_name
        )

        try:

            # ==================================
            # EXTRACT FEATURES
            # ==================================

            features = extract_features(
                file_path
            )

            if len(features) != len(feature_names):

                print(
                    f"\nFeature mismatch: {file_name}"
                )

                print(
                    f"Expected: {len(feature_names)}"
                )

                print(
                    f"Received: {len(features)}"
                )

                continue


            # ==================================
            # CREATE DATAFRAME
            # ==================================

            features_df = pd.DataFrame(
                [features],
                columns=feature_names
            )


            # ==================================
            # SCALE
            # ==================================

            features_scaled = scaler.transform(
                features_df
            )


            # ==================================
            # MODEL PROBABILITY
            # ==================================

            probabilities = model.predict_proba(
                features_scaled
            )[0]

            probability_map = dict(
                zip(
                    model.classes_,
                    probabilities
                )
            )

            real_probability = probability_map.get(
                0,
                0.0
            )

            fake_probability = probability_map.get(
                1,
                0.0
            )


            # ==================================
            # PREDICTION
            # ==================================

            predicted_label = int(
                fake_probability >= 0.50
            )


            # ==================================
            # RISK
            # ==================================

            risk_score = (
                fake_probability * 100
            )

            risk_level = get_risk_level(
                risk_score
            )


            # ==================================
            # PRINT RESULT
            # ==================================

            expected_text = (
                "REAL"
                if true_label == 0
                else "FAKE"
            )

            predicted_text = (
                "REAL"
                if predicted_label == 0
                else "FAKE"
            )

            print("\n" + "-" * 60)

            print(
                f"File {index}: {file_name}"
            )

            print(
                f"Expected      : {expected_text}"
            )

            print(
                f"Predicted     : {predicted_text}"
            )

            print(
                f"REAL Prob.    : {real_probability * 100:.2f}%"
            )

            print(
                f"FAKE Prob.    : {fake_probability * 100:.2f}%"
            )

            print(
                f"Risk Score    : {risk_score:.2f}%"
            )

            print(
                f"Risk Level    : {risk_level}"
            )


            # ==================================
            # STORE RESULT
            # ==================================

            results.append({
                "filename": file_name,
                "expected_label": true_label,
                "predicted_label": predicted_label,
                "real_probability": real_probability,
                "fake_probability": fake_probability,
                "risk_score": risk_score,
                "risk_level": risk_level
            })


        except Exception as error:

            print(
                f"\nERROR processing: {file_name}"
            )

            print(error)


    return results


# ==========================================
# MAIN EVALUATION
# ==========================================

def evaluate_system():

    print("\n" + "=" * 70)
    print("SWARRAKSHAK EXTERNAL EVALUATION")
    print("=" * 70)


    # ======================================
    # LOAD MODEL
    # ======================================

    print("\nLoading model and scaler...")

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    feature_names = get_feature_names()


    print(
        f"Expected feature count: {len(feature_names)}"
    )


    # ======================================
    # REAL EVALUATION
    # ======================================

    print("\n" + "=" * 70)
    print("EVALUATING REAL VOICES")
    print("=" * 70)

    real_results = process_folder(
        REAL_EVAL_FOLDER,
        0,
        model,
        scaler,
        feature_names
    )


    # ======================================
    # FAKE EVALUATION
    # ======================================

    print("\n" + "=" * 70)
    print("EVALUATING FAKE VOICES")
    print("=" * 70)

    fake_results = process_folder(
        FAKE_EVAL_FOLDER,
        1,
        model,
        scaler,
        feature_names
    )


    # ======================================
    # COMBINE RESULTS
    # ======================================

    results = (
        real_results
        +
        fake_results
    )


    if len(results) == 0:

        print(
            "\nNo evaluation files were processed."
        )

        return


    results_df = pd.DataFrame(
        results
    )


    y_true = results_df[
        "expected_label"
    ]

    y_pred = results_df[
        "predicted_label"
    ]


    # ======================================
    # METRICS
    # ======================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )


    tn, fp, fn, tp = cm.ravel()


    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )


    # ======================================
    # AVERAGE RISK
    # ======================================

    real_average_risk = results_df[
        results_df["expected_label"] == 0
    ]["risk_score"].mean()


    fake_average_risk = results_df[
        results_df["expected_label"] == 1
    ]["risk_score"].mean()


    # ======================================
    # PRINT FINAL REPORT
    # ======================================

    print("\n" + "=" * 70)
    print("FINAL EXTERNAL EVALUATION RESULT")
    print("=" * 70)


    print(
        f"\nTotal evaluated samples: {len(results_df)}"
    )

    print(
        f"REAL samples: {len(real_results)}"
    )

    print(
        f"FAKE samples: {len(fake_results)}"
    )


    print("\nPerformance:")

    print(
        f"Accuracy              : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision             : {precision * 100:.2f}%"
    )

    print(
        f"Recall (FAKE)         : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score              : {f1 * 100:.2f}%"
    )

    print(
        f"False Positive Rate   : {false_positive_rate * 100:.2f}%"
    )

    print(
        f"False Negative Rate   : {false_negative_rate * 100:.2f}%"
    )


    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )


    print(
        "\nMeaning:"
    )

    print(
        f"REAL correctly detected : {tn}"
    )

    print(
        f"REAL detected as FAKE   : {fp}"
    )

    print(
        f"FAKE detected as REAL   : {fn}"
    )

    print(
        f"FAKE correctly detected : {tp}"
    )


    print(
        "\nAverage Risk Scores:"
    )

    print(
        f"REAL average risk: {real_average_risk:.2f}%"
    )

    print(
        f"FAKE average risk: {fake_average_risk:.2f}%"
    )


    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "REAL",
                "FAKE"
            ],
            zero_division=0
        )
    )


    # ======================================
    # SAVE RESULTS
    # ======================================

    os.makedirs(
        os.path.dirname(
            OUTPUT_CSV
        ),
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )


    print(
        f"\nDetailed results saved to:"
    )

    print(
        OUTPUT_CSV
    )


    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    evaluate_system()