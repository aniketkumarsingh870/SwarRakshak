import os
import joblib
import librosa
import numpy as np
import pandas as pd


# ==========================================
# SWARRAKSHAK
# STORED AUDIO MODEL TEST
# ==========================================


MODEL_PATH = "models/swarrakshak_model.pkl"
SCALER_PATH = "models/swarrakshak_scaler.pkl"

AUDIO_FILE = "data/samples/test_audio.wav"

SAMPLE_RATE = 16000
N_MFCC = 13


FEATURE_NAMES = [

    *[
        f"mfcc_mean_{i + 1}"
        for i in range(N_MFCC)
    ],

    *[
        f"mfcc_std_{i + 1}"
        for i in range(N_MFCC)
    ],

    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_rolloff",
    "rms_energy"
]


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate
    )

    rms = librosa.feature.rms(
        y=audio
    )

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            [
                np.mean(zcr),
                np.mean(spectral_centroid),
                np.mean(spectral_rolloff),
                np.mean(rms)
            ]
        ]
    )

    return features


# ==========================================
# RISK LEVEL
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
# TEST AUDIO
# ==========================================

def analyze_audio():

    print("\n")
    print("=" * 60)
    print("SWARRAKSHAK STORED AUDIO ANALYSIS")
    print("=" * 60)

    if not os.path.exists(AUDIO_FILE):

        print(
            f"\nAudio file not found: {AUDIO_FILE}"
        )

        return

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    features = extract_features(
        AUDIO_FILE
    )

    features_df = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES
    )

    features_scaled = scaler.transform(
        features_df
    )

    probabilities = model.predict_proba(
        features_scaled
    )[0]

    class_probabilities = dict(
        zip(
            model.classes_,
            probabilities
        )
    )

    real_probability = float(
        class_probabilities.get(
            0,
            0.0
        )
    )

    fake_probability = float(
        class_probabilities.get(
            1,
            0.0
        )
    )

    risk_score = (
        fake_probability * 100
    )

    risk_level = get_risk_level(
        risk_score
    )

    print(
        f"\nREAL Confidence : "
        f"{real_probability * 100:.2f}%"
    )

    print(
        f"FAKE Confidence : "
        f"{fake_probability * 100:.2f}%"
    )

    print(
        f"Risk Score      : "
        f"{risk_score:.2f}%"
    )

    print(
        f"Risk Level      : "
        f"{risk_level}"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":

    analyze_audio()