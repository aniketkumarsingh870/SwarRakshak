import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# PATHS
# ==========================================

DATA_FILE = "data/processed/voice_features.csv"

MODEL_PATH = "models/swarrakshak_model.pkl"
SCALER_PATH = "models/swarrakshak_scaler.pkl"


# ==========================================
# LOAD DATASET
# ==========================================

print("\n" + "=" * 65)
print("SWARRAKSHAK MODEL TRAINING")
print("=" * 65)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Total samples: {len(df)}")


# ==========================================
# PREPARE X AND Y
# ==========================================

drop_columns = [
    "label",
    "source",
    "filename"
]

X = df.drop(
    columns=[
        column
        for column in drop_columns
        if column in df.columns
    ]
)

y = df["label"]


print(
    f"Number of features: {X.shape[1]}"
)

print(
    "\nClass distribution:"
)

print(
    y.value_counts()
)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ==========================================
# STANDARD SCALER
# ==========================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================
# MODEL 1: RANDOM FOREST
# ==========================================

print("\n" + "-" * 65)
print("Training Random Forest...")
print("-" * 65)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_scaled,
    y_train
)

rf_prediction = rf_model.predict(
    X_test_scaled
)

rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)

rf_precision = precision_score(
    y_test,
    rf_prediction,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_prediction,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_prediction,
    zero_division=0
)


print(
    f"Accuracy : {rf_accuracy * 100:.2f}%"
)

print(
    f"Precision: {rf_precision * 100:.2f}%"
)

print(
    f"Recall   : {rf_recall * 100:.2f}%"
)

print(
    f"F1 Score : {rf_f1 * 100:.2f}%"
)


# ==========================================
# MODEL 2: SVM
# ==========================================

print("\n" + "-" * 65)
print("Training SVM...")
print("-" * 65)

svm_model = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    C=2.0,
    gamma="scale",
    random_state=42
)

svm_model.fit(
    X_train_scaled,
    y_train
)

svm_prediction = svm_model.predict(
    X_test_scaled
)

svm_accuracy = accuracy_score(
    y_test,
    svm_prediction
)

svm_precision = precision_score(
    y_test,
    svm_prediction,
    zero_division=0
)

svm_recall = recall_score(
    y_test,
    svm_prediction,
    zero_division=0
)

svm_f1 = f1_score(
    y_test,
    svm_prediction,
    zero_division=0
)


print(
    f"Accuracy : {svm_accuracy * 100:.2f}%"
)

print(
    f"Precision: {svm_precision * 100:.2f}%"
)

print(
    f"Recall   : {svm_recall * 100:.2f}%"
)

print(
    f"F1 Score : {svm_f1 * 100:.2f}%"
)


# ==========================================
# SELECT BEST MODEL BY F1 SCORE
# ==========================================

if svm_f1 > rf_f1:

    best_model = svm_model

    best_name = "SVM"

    best_prediction = svm_prediction

    best_f1 = svm_f1

else:

    best_model = rf_model

    best_name = "Random Forest"

    best_prediction = rf_prediction

    best_f1 = rf_f1


# ==========================================
# BEST MODEL REPORT
# ==========================================

print("\n" + "=" * 65)
print("BEST MODEL")
print("=" * 65)

print(
    f"\nSelected Model: {best_name}"
)

print(
    f"F1 Score: {best_f1 * 100:.2f}%"
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        best_prediction
    )
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        best_prediction,
        target_names=[
            "REAL",
            "FAKE"
        ],
        zero_division=0
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)


print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)

print(
    f"\nModel saved to: {MODEL_PATH}"
)

print(
    f"Scaler saved to: {SCALER_PATH}"
)

print(
    f"Feature count: {X.shape[1]}"
)

print(
    "\nIMPORTANT:"
)

print(
    "External evaluation data was NOT used for training."
)