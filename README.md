# SwarRakshak

## AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

SwarRakshak is a prototype AI-based voice security system designed to detect possible AI-generated, cloned, or spoofed speech in real time.

The system analyzes microphone audio, extracts acoustic features, uses a machine-learning classifier to estimate whether the voice is genuine or synthetic, and then calculates a dynamic impersonation risk score.

---

## Project Objective

The main objective of SwarRakshak is to help detect voice cloning impersonation attacks during live voice communication.

Instead of producing only a REAL or FAKE output, the system also provides a dynamic risk level:

- LOW
- MEDIUM
- HIGH
- CRITICAL

The dynamic risk score considers both the current machine-learning prediction and recent suspicious voice activity.

---

## Current Prototype Features

- Real-time microphone recording
- 16 kHz mono audio processing
- AI-generated voice detection
- Genuine voice detection
- 164-dimensional acoustic feature extraction
- MFCC features
- Delta MFCC
- Delta-Delta MFCC
- Log-Mel spectral features
- Zero Crossing Rate
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Spectral Flatness
- RMS Energy
- Random Forest and SVM model training
- Dynamic risk scoring
- Temporal risk smoothing
- Suspicious-window persistence tracking
- LOW / MEDIUM / HIGH / CRITICAL risk levels
- Real-time warning alerts
- Streamlit dashboard
- Detection history
- Privacy-aware RAM-only live audio processing

---

## System Architecture

```text
Microphone / Audio Input
        |
        v
Audio Preprocessing
        |
        v
Feature Extraction
        |
        |-- MFCC
        |-- Delta MFCC
        |-- Delta-Delta MFCC
        |-- Log-Mel Features
        |-- Spectral Features
        |
        v
Machine Learning Model
        |
        v
REAL / FAKE Probability
        |
        v
Dynamic Risk Engine
        |
        |-- Current ML Risk
        |-- Temporal Risk
        |-- Suspicious Persistence
        |-- Consecutive Suspicious Windows
        |
        v
Final Risk Score
        |
        v
LOW / MEDIUM / HIGH / CRITICAL
        |
        v
Streamlit Dashboard + Alert