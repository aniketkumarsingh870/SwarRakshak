import time
from collections import deque

import joblib
import numpy as np
import pandas as pd
import sounddevice as sd

from feature_extractor import (
    extract_features_from_audio,
    get_feature_names
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

WINDOW_SECONDS = 3

SAMPLES_PER_WINDOW = (
    SAMPLE_RATE * WINDOW_SECONDS
)

MODEL_PATH = "models/swarrakshak_model.pkl"
SCALER_PATH = "models/swarrakshak_scaler.pkl"

SILENCE_THRESHOLD = 0.005

HISTORY_SIZE = 5

# How much recent history affects temporal risk
EMA_ALPHA = 0.35


# ============================================================
# HISTORY
# ============================================================

risk_history = deque(
    maxlen=HISTORY_SIZE
)

prediction_history = deque(
    maxlen=HISTORY_SIZE
)

ema_risk = None

previous_risk = None

consecutive_suspicious = 0


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(score):

    if score < 30:
        return "LOW"

    elif score < 55:
        return "MEDIUM"

    elif score < 80:
        return "HIGH"

    else:
        return "CRITICAL"


# ============================================================
# SYSTEM STATUS
# ============================================================

def get_system_status(risk_level):

    if risk_level == "LOW":
        return "LIKELY GENUINE"

    elif risk_level == "MEDIUM":
        return "UNCERTAIN / MONITOR"

    elif risk_level == "HIGH":
        return "SUSPICIOUS VOICE"

    else:
        return "POSSIBLE IMPERSONATION ATTACK"


# ============================================================
# ALERT
# ============================================================

def show_alert(risk_level):

    if risk_level == "HIGH":

        print(
            "\n[WARNING] Suspicious voice characteristics detected."
        )

    elif risk_level == "CRITICAL":

        print("\n" + "!" * 70)

        print(
            "CRITICAL ALERT: POSSIBLE VOICE IMPERSONATION ATTACK"
        )

        print(
            "Do not trust sensitive instructions without verification."
        )

        print("!" * 70)


# ============================================================
# DYNAMIC RISK ENGINE
# ============================================================

def calculate_dynamic_risk(fake_probability):

    global ema_risk
    global previous_risk
    global consecutive_suspicious

    # --------------------------------------------------------
    # 1. CURRENT ML SPOOF SCORE
    # --------------------------------------------------------

    current_risk = (
        fake_probability * 100
    )


    # --------------------------------------------------------
    # 2. TEMPORAL EMA
    # --------------------------------------------------------

    if ema_risk is None:

        ema_risk = current_risk

    else:

        ema_risk = (
            EMA_ALPHA * current_risk
            +
            (1 - EMA_ALPHA) * ema_risk
        )


    # --------------------------------------------------------
    # 3. STORE HISTORY
    # --------------------------------------------------------

    risk_history.append(
        current_risk
    )


    is_suspicious = (
        current_risk >= 50
    )

    prediction_history.append(
        1 if is_suspicious else 0
    )


    # --------------------------------------------------------
    # 4. CONSECUTIVE SUSPICIOUS WINDOWS
    # --------------------------------------------------------

    if is_suspicious:

        consecutive_suspicious += 1

    else:

        consecutive_suspicious = 0


    # --------------------------------------------------------
    # 5. PERSISTENCE SCORE
    # --------------------------------------------------------

    if len(prediction_history) > 0:

        persistence_score = (
            sum(prediction_history)
            /
            len(prediction_history)
        ) * 100

    else:

        persistence_score = 0


    # --------------------------------------------------------
    # 6. RISK TREND
    # --------------------------------------------------------

    trend_bonus = 0

    if previous_risk is not None:

        increase = (
            current_risk - previous_risk
        )

        if increase > 0:

            trend_bonus = min(
                increase * 0.30,
                8
            )


    previous_risk = current_risk


    # --------------------------------------------------------
    # 7. BASE DYNAMIC RISK
    # --------------------------------------------------------

    dynamic_risk = (

        0.50 * current_risk

        +

        0.30 * ema_risk

        +

        0.20 * persistence_score
    )


    # --------------------------------------------------------
    # 8. RAPID-INCREASE BONUS
    # --------------------------------------------------------

    dynamic_risk += trend_bonus


    # --------------------------------------------------------
    # 9. PERSISTENT ATTACK BONUS
    # --------------------------------------------------------

    if consecutive_suspicious >= 3:

        dynamic_risk += 5


    if consecutive_suspicious >= 5:

        dynamic_risk += 5


    # --------------------------------------------------------
    # KEEP BETWEEN 0 AND 100
    # --------------------------------------------------------

    dynamic_risk = np.clip(
        dynamic_risk,
        0,
        100
    )


    return {
        "current_risk": current_risk,
        "ema_risk": ema_risk,
        "persistence_score": persistence_score,
        "trend_bonus": trend_bonus,
        "consecutive_suspicious": consecutive_suspicious,
        "dynamic_risk": dynamic_risk
    }


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)

print(
    "SWARRAKSHAK - DYNAMIC REAL-TIME VOICE SECURITY SYSTEM"
)

print("=" * 70)

print("\nLoading trained model...")


model = joblib.load(
    MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

feature_names = get_feature_names()


print(
    f"Feature count: {len(feature_names)}"
)

print(
    "Model loaded successfully."
)


# ============================================================
# ANALYZE AUDIO
# ============================================================

def analyze_audio(audio):

    # --------------------------------------------------------
    # SILENCE CHECK
    # --------------------------------------------------------

    rms = np.sqrt(
        np.mean(
            np.square(audio)
        )
    )


    if rms < SILENCE_THRESHOLD:

        print(
            "\nNo clear speech detected."
        )

        return


    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    features = extract_features_from_audio(
        audio,
        SAMPLE_RATE
    )


    if len(features) != len(feature_names):

        print(
            "\nERROR: Feature count mismatch."
        )

        print(
            f"Expected: {len(feature_names)}"
        )

        print(
            f"Received: {len(features)}"
        )

        return


    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        [features],
        columns=feature_names
    )


    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    scaled_features = scaler.transform(
        feature_df
    )


    # --------------------------------------------------------
    # MODEL PROBABILITY
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        scaled_features
    )[0]


    probability_map = dict(
        zip(
            model.classes_,
            probabilities
        )
    )


    real_probability = probability_map.get(
        0,
        0
    )

    fake_probability = probability_map.get(
        1,
        0
    )


    # --------------------------------------------------------
    # RAW MODEL CLASSIFICATION
    # --------------------------------------------------------

    if fake_probability >= 0.50:

        model_prediction = "FAKE"

    else:

        model_prediction = "REAL"


    # --------------------------------------------------------
    # DYNAMIC RISK
    # --------------------------------------------------------

    risk_data = calculate_dynamic_risk(
        fake_probability
    )


    current_risk = risk_data[
        "current_risk"
    ]

    temporal_risk = risk_data[
        "ema_risk"
    ]

    persistence = risk_data[
        "persistence_score"
    ]

    consecutive = risk_data[
        "consecutive_suspicious"
    ]

    dynamic_risk = risk_data[
        "dynamic_risk"
    ]


    # --------------------------------------------------------
    # FINAL RISK LEVEL
    # --------------------------------------------------------

    risk_level = get_risk_level(
        dynamic_risk
    )


    system_status = get_system_status(
        risk_level
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "SWARRAKSHAK LIVE SECURITY ANALYSIS"
    )

    print("=" * 70)


    print(
        f"REAL Probability        : "
        f"{real_probability * 100:.2f}%"
    )

    print(
        f"FAKE Probability        : "
        f"{fake_probability * 100:.2f}%"
    )


    print(
        f"\nML Classification       : "
        f"{model_prediction}"
    )


    print("\n--- Dynamic Risk Analysis ---")


    print(
        f"Current ML Risk         : "
        f"{current_risk:.2f}%"
    )

    print(
        f"Temporal Risk           : "
        f"{temporal_risk:.2f}%"
    )

    print(
        f"Suspicious Persistence  : "
        f"{persistence:.2f}%"
    )

    print(
        f"Consecutive Suspicious  : "
        f"{consecutive}"
    )


    print(
        f"\nFINAL DYNAMIC RISK      : "
        f"{dynamic_risk:.2f}%"
    )

    print(
        f"RISK LEVEL              : "
        f"{risk_level}"
    )

    print(
        f"SYSTEM STATUS           : "
        f"{system_status}"
    )


    show_alert(
        risk_level
    )


    print("=" * 70)


# ============================================================
# REAL-TIME LOOP
# ============================================================

def start_realtime_detection():

    print("\n" + "=" * 70)

    print(
        "REAL-TIME PROTECTION STARTED"
    )

    print("=" * 70)


    print(
        f"\nAudio Window: {WINDOW_SECONDS} seconds"
    )

    print(
        f"Risk History: {HISTORY_SIZE} windows"
    )

    print(
        "Audio Storage: RAM only"
    )

    print(
        "Raw microphone audio is NOT saved."
    )

    print(
        "\nPress Ctrl + C to stop."
    )


    while True:

        try:

            print(
                "\nListening..."
            )


            recording = sd.rec(
                SAMPLES_PER_WINDOW,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )


            sd.wait()


            audio = recording.flatten()


            analyze_audio(
                audio
            )


            # Release the audio reference
            del audio
            del recording


            time.sleep(
                0.15
            )


        except KeyboardInterrupt:

            print("\n" + "=" * 70)

            print(
                "SWARRAKSHAK REAL-TIME PROTECTION STOPPED"
            )

            print("=" * 70)

            break


        except Exception as error:

            print(
                "\nREAL-TIME ANALYSIS ERROR:"
            )

            print(
                error
            )

            time.sleep(
                1
            )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    start_realtime_detection()