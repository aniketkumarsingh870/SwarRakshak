import os
import sys
import time
from collections import deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sounddevice as sd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


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

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "swarrakshak_model.pkl"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "swarrakshak_scaler.pkl"
)

LOG_PATH = os.path.join(
    PROJECT_ROOT,
    "logs",
    "detection_log.csv"
)

SILENCE_THRESHOLD = 0.005

HISTORY_SIZE = 5

EMA_ALPHA = 0.35


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SwarRakshak",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_system():

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    feature_names = get_feature_names()

    return (
        model,
        scaler,
        feature_names
    )


model, scaler, feature_names = load_system()


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "monitoring": False,
    "ema_risk": None,
    "previous_risk": None,
    "consecutive_suspicious": 0,
    "risk_history": deque(maxlen=HISTORY_SIZE),
    "prediction_history": deque(maxlen=HISTORY_SIZE),
    "history_table": [],
    "latest_result": None
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# RISK FUNCTIONS
# ============================================================

def get_risk_level(score):

    if score < 30:
        return "LOW"

    elif score < 55:
        return "MEDIUM"

    elif score < 80:
        return "HIGH"

    return "CRITICAL"


def get_status(level):

    if level == "LOW":
        return "LIKELY GENUINE"

    elif level == "MEDIUM":
        return "UNCERTAIN / MONITOR"

    elif level == "HIGH":
        return "SUSPICIOUS VOICE"

    return "POSSIBLE IMPERSONATION ATTACK"


# ============================================================
# DYNAMIC RISK
# ============================================================

def calculate_dynamic_risk(fake_probability):

    current_risk = (
        fake_probability * 100
    )

    # EMA
    if st.session_state.ema_risk is None:

        st.session_state.ema_risk = (
            current_risk
        )

    else:

        st.session_state.ema_risk = (

            EMA_ALPHA
            * current_risk

            +

            (1 - EMA_ALPHA)
            * st.session_state.ema_risk
        )


    temporal_risk = (
        st.session_state.ema_risk
    )


    # Store risk
    st.session_state.risk_history.append(
        current_risk
    )


    suspicious = (
        current_risk >= 50
    )


    st.session_state.prediction_history.append(
        1 if suspicious else 0
    )


    # Consecutive suspicious
    if suspicious:

        st.session_state.consecutive_suspicious += 1

    else:

        st.session_state.consecutive_suspicious = 0


    # Persistence
    persistence = (

        sum(
            st.session_state.prediction_history
        )

        /

        len(
            st.session_state.prediction_history
        )

    ) * 100


    # Trend
    trend_bonus = 0


    if st.session_state.previous_risk is not None:

        increase = (

            current_risk
            -
            st.session_state.previous_risk
        )


        if increase > 0:

            trend_bonus = min(
                increase * 0.30,
                8
            )


    st.session_state.previous_risk = (
        current_risk
    )


    # Final dynamic risk
    dynamic_risk = (

        0.50 * current_risk

        +

        0.30 * temporal_risk

        +

        0.20 * persistence

    )


    dynamic_risk += trend_bonus


    if (
        st.session_state.consecutive_suspicious
        >= 3
    ):

        dynamic_risk += 5


    if (
        st.session_state.consecutive_suspicious
        >= 5
    ):

        dynamic_risk += 5


    dynamic_risk = float(
        np.clip(
            dynamic_risk,
            0,
            100
        )
    )


    return {

        "current_risk":
            current_risk,

        "temporal_risk":
            temporal_risk,

        "persistence":
            persistence,

        "consecutive":
            st.session_state.consecutive_suspicious,

        "dynamic_risk":
            dynamic_risk

    }


# ============================================================
# ANALYZE AUDIO
# ============================================================

def analyze_audio(audio):

    rms = np.sqrt(
        np.mean(
            np.square(audio)
        )
    )


    if rms < SILENCE_THRESHOLD:

        return {
            "error":
                "No clear speech detected."
        }


    features = extract_features_from_audio(
        audio,
        SAMPLE_RATE
    )


    features_df = pd.DataFrame(
        [features],
        columns=feature_names
    )


    scaled_features = scaler.transform(
        features_df
    )


    probabilities = model.predict_proba(
        scaled_features
    )[0]


    probability_map = dict(
        zip(
            model.classes_,
            probabilities
        )
    )


    real_probability = float(
        probability_map.get(
            0,
            0
        )
    )


    fake_probability = float(
        probability_map.get(
            1,
            0
        )
    )


    risk_data = calculate_dynamic_risk(
        fake_probability
    )


    risk_level = get_risk_level(
        risk_data[
            "dynamic_risk"
        ]
    )


    status = get_status(
        risk_level
    )


    prediction = (

        "FAKE"

        if fake_probability >= 0.50

        else

        "REAL"

    )


    return {

        "real_probability":
            real_probability,

        "fake_probability":
            fake_probability,

        "prediction":
            prediction,

        "risk_level":
            risk_level,

        "status":
            status,

        **risk_data

    }


# ============================================================
# LOGGING
# ============================================================

def save_log(result):

    os.makedirs(
        os.path.dirname(LOG_PATH),
        exist_ok=True
    )


    row = pd.DataFrame([
        {

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "real_probability":
                result[
                    "real_probability"
                ],

            "fake_probability":
                result[
                    "fake_probability"
                ],

            "current_risk":
                result[
                    "current_risk"
                ],

            "dynamic_risk":
                result[
                    "dynamic_risk"
                ],

            "risk_level":
                result[
                    "risk_level"
                ],

            "prediction":
                result[
                    "prediction"
                ]
        }
    ])


    if os.path.exists(LOG_PATH):

        row.to_csv(
            LOG_PATH,
            mode="a",
            header=False,
            index=False
        )

    else:

        row.to_csv(
            LOG_PATH,
            index=False
        )


# ============================================================
# RESET
# ============================================================

def reset_system():

    st.session_state.ema_risk = None

    st.session_state.previous_risk = None

    st.session_state.consecutive_suspicious = 0

    st.session_state.risk_history = deque(
        maxlen=HISTORY_SIZE
    )

    st.session_state.prediction_history = deque(
        maxlen=HISTORY_SIZE
    )

    st.session_state.history_table = []

    st.session_state.latest_result = None


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ SwarRakshak"
)

st.caption(
    "AI-Powered Real-Time Detection and Prevention "
    "of Voice Cloning Impersonation Attacks"
)


st.divider()


# ============================================================
# SYSTEM INFO
# ============================================================

a, b, c, d = st.columns(4)


with a:

    st.metric(
        "Model Features",
        len(feature_names)
    )


with b:

    st.metric(
        "Analysis Window",
        f"{WINDOW_SECONDS} sec"
    )


with c:

    status_text = (

        "ACTIVE"

        if st.session_state.monitoring

        else

        "STOPPED"
    )

    st.metric(
        "Protection Status",
        status_text
    )


with d:

    st.metric(
        "Audio Storage",
        "RAM Only"
    )


st.divider()


# ============================================================
# CONTROLS
# ============================================================

st.subheader(
    "Real-Time Protection"
)


c1, c2, c3 = st.columns(3)


with c1:

    if st.button(
        "▶ Start Monitoring",
        use_container_width=True,
        type="primary"
    ):

        reset_system()

        st.session_state.monitoring = True

        st.rerun()


with c2:

    if st.button(
        "■ Stop Monitoring",
        use_container_width=True
    ):

        st.session_state.monitoring = False

        st.rerun()


with c3:

    if st.button(
        "↻ Reset Session",
        use_container_width=True
    ):

        st.session_state.monitoring = False

        reset_system()

        st.rerun()


# ============================================================
# STATUS MESSAGE
# ============================================================

if st.session_state.monitoring:

    st.success(
        "Protection active — microphone is being monitored."
    )

else:

    st.info(
        "Monitoring is currently stopped."
    )


# ============================================================
# LIVE RESULT PLACEHOLDER
# ============================================================

result_container = st.empty()


# ============================================================
# HISTORY
# ============================================================

st.divider()

st.subheader(
    "Recent Detection History"
)


history_container = st.empty()


# ============================================================
# PRIVACY
# ============================================================

st.divider()

st.caption(
    "Privacy: Microphone audio is processed temporarily "
    "in memory and is not stored. Only detection metadata "
    "is written to the local detection log."
)


# ============================================================
# CONTINUOUS MONITORING
# ============================================================

if st.session_state.monitoring:

    try:

        with st.spinner(
            "Listening..."
        ):

            recording = sd.rec(
                SAMPLES_PER_WINDOW,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )

            sd.wait()

            audio = recording.flatten()


        result = analyze_audio(
            audio
        )


        del audio
        del recording


        if "error" not in result:

            st.session_state.latest_result = (
                result
            )


            save_log(
                result
            )


            history_row = {

                "Time":
                    datetime.now().strftime(
                        "%H:%M:%S"
                    ),

                "REAL %":
                    round(
                        result[
                            "real_probability"
                        ] * 100,
                        2
                    ),

                "FAKE %":
                    round(
                        result[
                            "fake_probability"
                        ] * 100,
                        2
                    ),

                "Risk %":
                    round(
                        result[
                            "dynamic_risk"
                        ],
                        2
                    ),

                "Level":
                    result[
                        "risk_level"
                    ],

                "Prediction":
                    result[
                        "prediction"
                    ]

            }


            st.session_state.history_table.insert(
                0,
                history_row
            )


            st.session_state.history_table = (
                st.session_state.history_table[:20]
            )


        else:

            st.warning(
                result["error"]
            )


        time.sleep(
            0.2
        )


        st.rerun()


    except Exception as error:

        st.session_state.monitoring = False

        st.error(
            f"Monitoring error: {error}"
        )


# ============================================================
# DISPLAY LATEST RESULT
# ============================================================

latest = (
    st.session_state.latest_result
)


if latest is not None:

    with result_container.container():

        st.subheader(
            "Live Security Analysis"
        )


        m1, m2, m3, m4 = st.columns(4)


        m1.metric(
            "REAL Probability",
            f"{latest['real_probability'] * 100:.2f}%"
        )


        m2.metric(
            "FAKE Probability",
            f"{latest['fake_probability'] * 100:.2f}%"
        )


        m3.metric(
            "Dynamic Risk",
            f"{latest['dynamic_risk']:.2f}%"
        )


        m4.metric(
            "Risk Level",
            latest[
                "risk_level"
            ]
        )


        st.progress(
            int(
                latest[
                    "dynamic_risk"
                ]
            )
        )


        st.write(
            "**ML Classification:** "
            + latest[
                "prediction"
            ]
        )


        st.write(
            "**System Status:** "
            + latest[
                "status"
            ]
        )


        st.write(
            f"**Suspicious Persistence:** "
            f"{latest['persistence']:.2f}%"
        )


        st.write(
            f"**Consecutive Suspicious Windows:** "
            f"{latest['consecutive']}"
        )


        if latest["risk_level"] == "LOW":

            st.success(
                "Voice currently appears likely genuine."
            )


        elif latest["risk_level"] == "MEDIUM":

            st.warning(
                "Uncertain voice pattern. Continue monitoring."
            )


        elif latest["risk_level"] == "HIGH":

            st.error(
                "Suspicious voice characteristics detected."
            )


        else:

            st.error(
                "CRITICAL: Possible voice impersonation attack."
            )


# ============================================================
# DISPLAY HISTORY
# ============================================================

if len(
    st.session_state.history_table
) > 0:

    history_df = pd.DataFrame(
        st.session_state.history_table
    )


    history_container.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


else:

    history_container.info(
        "No detection history yet."
    )