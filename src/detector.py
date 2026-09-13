import numpy as np


def analyze_voice(features):

    # Convert features to NumPy array
    features = np.array(features)

    # Calculate statistical information
    feature_mean = np.mean(features)
    feature_std = np.std(features)

    # ------------------------------------------------
    # TEMPORARY DETECTION LOGIC
    # ------------------------------------------------
    # This is NOT the final AI model.
    # Later, we will replace this with a trained
    # Machine Learning / Deep Learning model.
    # ------------------------------------------------

    raw_score = abs(feature_mean) + feature_std

    # Convert the score into a percentage
    risk_score = min(raw_score * 10, 100)

    # Determine risk level
    if risk_score < 30:
        risk_level = "LOW"

    elif risk_score < 60:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    return risk_score, risk_level