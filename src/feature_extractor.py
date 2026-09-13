import librosa
import numpy as np


SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 40


def extract_features_from_audio(audio, sr=SAMPLE_RATE):

    audio = np.asarray(audio, dtype=np.float32).flatten()

    if sr != SAMPLE_RATE:

        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

        sr = SAMPLE_RATE

    # Remove silence from beginning/end
    audio, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    if len(audio) == 0:
        raise ValueError("Audio is empty after trimming.")

    # ======================================
    # MFCC
    # ======================================

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # ======================================
    # DELTA MFCC
    # ======================================

    delta = librosa.feature.delta(
        mfcc,
        order=1
    )

    delta_mean = np.mean(
        delta,
        axis=1
    )

    delta_std = np.std(
        delta,
        axis=1
    )

    # ======================================
    # DELTA-DELTA MFCC
    # ======================================

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    delta2_mean = np.mean(
        delta2,
        axis=1
    )

    delta2_std = np.std(
        delta2,
        axis=1
    )

    # ======================================
    # LOG MEL
    # ======================================

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        power=2.0
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    log_mel_mean = np.mean(
        log_mel,
        axis=1
    )

    log_mel_std = np.std(
        log_mel,
        axis=1
    )

    # ======================================
    # OTHER FEATURES
    # ======================================

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(
            audio
        )
    )

    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )
    )

    bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )
    )

    rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr
        )
    )

    flatness = np.mean(
        librosa.feature.spectral_flatness(
            y=audio
        )
    )

    rms = np.mean(
        librosa.feature.rms(
            y=audio
        )
    )

    # ======================================
    # COMBINE
    # ======================================

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,

        delta_mean,
        delta_std,

        delta2_mean,
        delta2_std,

        log_mel_mean,
        log_mel_std,

        [
            zcr,
            centroid,
            bandwidth,
            rolloff,
            flatness,
            rms
        ]
    ])

    return features


def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    return extract_features_from_audio(
        audio,
        sr
    )


def get_feature_names():

    names = []

    names.extend([
        f"mfcc_mean_{i + 1}"
        for i in range(N_MFCC)
    ])

    names.extend([
        f"mfcc_std_{i + 1}"
        for i in range(N_MFCC)
    ])

    names.extend([
        f"delta_mean_{i + 1}"
        for i in range(N_MFCC)
    ])

    names.extend([
        f"delta_std_{i + 1}"
        for i in range(N_MFCC)
    ])

    names.extend([
        f"delta2_mean_{i + 1}"
        for i in range(N_MFCC)
    ])

    names.extend([
        f"delta2_std_{i + 1}"
        for i in range(N_MFCC)
    ])

    names.extend([
        f"logmel_mean_{i + 1}"
        for i in range(N_MELS)
    ])

    names.extend([
        f"logmel_std_{i + 1}"
        for i in range(N_MELS)
    ])

    names.extend([
        "zero_crossing_rate",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_flatness",
        "rms_energy"
    ])

    return names


if __name__ == "__main__":

    test_file = "data/samples/test_audio.wav"

    features = extract_features(
        test_file
    )

    names = get_feature_names()

    print("\n" + "=" * 60)
    print("SWARRAKSHAK FEATURE EXTRACTOR TEST")
    print("=" * 60)

    print(
        f"\nFeature count: {len(features)}"
    )

    print(
        f"Feature names: {len(names)}"
    )

    if len(features) == 164 and len(names) == 164:

        print(
            "\nFeature extractor working correctly."
        )

    else:

        print(
            "\nERROR: Expected 164 features."
        )