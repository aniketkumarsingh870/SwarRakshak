import os
import random
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf


# ==========================================
# SWARRAKSHAK
# LIVE FAKE / REPLAY SAMPLE COLLECTOR
# ==========================================


SOURCE_FOLDER = "data/dataset/fake"
OUTPUT_FOLDER = "data/live_fake"

SAMPLE_RATE = 16000
DURATION = 5

SAMPLES_PER_RECORDING = SAMPLE_RATE * DURATION

MIN_RMS = 0.003


# ==========================================
# RMS
# ==========================================

def calculate_rms(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    return np.sqrt(
        np.mean(
            np.square(audio)
        )
    )


# ==========================================
# NEXT FILE INDEX
# ==========================================

def get_next_index():

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    files = [

        file
        for file in os.listdir(OUTPUT_FOLDER)

        if file.lower().endswith(".wav")
    ]

    indices = []

    for file in files:

        try:

            number = int(
                file
                .replace("live_fake_", "")
                .replace(".wav", "")
            )

            indices.append(
                number
            )

        except ValueError:

            pass

    if not indices:

        return 0

    return max(indices) + 1


# ==========================================
# LOAD FAKE AUDIO
# ==========================================

def load_fake_audio(file_path):

    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # ----------------------------------
    # FIX LENGTH TO 5 SECONDS
    # ----------------------------------

    if len(audio) > SAMPLES_PER_RECORDING:

        audio = audio[
            :SAMPLES_PER_RECORDING
        ]

    elif len(audio) < SAMPLES_PER_RECORDING:

        padding = (
            SAMPLES_PER_RECORDING
            - len(audio)
        )

        audio = np.pad(
            audio,
            (0, padding)
        )

    return audio.astype(
        np.float32
    )


# ==========================================
# COLLECT LIVE FAKE DATA
# ==========================================

def collect_live_fake():

    print("\n")
    print("=" * 65)
    print("SWARRAKSHAK LIVE FAKE SAMPLE COLLECTION")
    print("=" * 65)

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    if not os.path.exists(
        SOURCE_FOLDER
    ):

        print(
            f"\nFake dataset folder not found: {SOURCE_FOLDER}"
        )

        return

    fake_files = [

        file
        for file in os.listdir(
            SOURCE_FOLDER
        )

        if file.lower().endswith(".wav")
    ]

    if len(fake_files) == 0:

        print(
            "\nNo fake WAV files found."
        )

        return

    print(
        f"\nAvailable fake source files: {len(fake_files)}"
    )

    try:

        number_of_samples = int(
            input(
                "\nHow many live fake samples do you want to record? "
            )
        )

    except ValueError:

        print(
            "\nInvalid number."
        )

        return

    if number_of_samples <= 0:

        print(
            "\nNumber must be greater than 0."
        )

        return

    if number_of_samples > len(fake_files):

        number_of_samples = len(
            fake_files
        )

    # Random source selection

    selected_files = random.sample(
        fake_files,
        number_of_samples
    )

    current_index = get_next_index()

    saved_count = 0

    print("\nIMPORTANT:")
    print(
        "The fake audio will play through your speakers."
    )
    print(
        "Your microphone will record the played audio."
    )
    print(
        "Keep your laptop speaker volume around 40-60%."
    )
    print(
        "Do not speak while samples are being recorded."
    )

    input(
        "\nPress ENTER when ready..."
    )

    for source_file in selected_files:

        print("\n" + "-" * 65)

        print(
            f"Sample {saved_count + 1}/{number_of_samples}"
        )

        source_path = os.path.join(
            SOURCE_FOLDER,
            source_file
        )

        print(
            f"Source: {source_file}"
        )

        fake_audio = load_fake_audio(
            source_path
        )

        print(
            "Playing fake audio and recording microphone..."
        )

        # ----------------------------------
        # PLAY THROUGH SPEAKER +
        # RECORD MICROPHONE SIMULTANEOUSLY
        # ----------------------------------

        recorded_audio = sd.playrec(
            fake_audio,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        recorded_audio = (
            recorded_audio
            .flatten()
        )

        rms = calculate_rms(
            recorded_audio
        )

        print(
            f"Recorded RMS: {rms:.6f}"
        )

        # ----------------------------------
        # REJECT VERY QUIET RECORDING
        # ----------------------------------

        if rms < MIN_RMS:

            print(
                "Recording too quiet — skipped."
            )

            print(
                "Increase speaker volume slightly."
            )

            continue

        output_file = os.path.join(
            OUTPUT_FOLDER,
            f"live_fake_{current_index:04d}.wav"
        )

        sf.write(
            output_file,
            recorded_audio,
            SAMPLE_RATE
        )

        print(
            f"Saved: {output_file}"
        )

        current_index += 1
        saved_count += 1

    print("\n")
    print("=" * 65)
    print("LIVE FAKE COLLECTION COMPLETE")
    print("=" * 65)

    print(
        f"\nNew samples saved: {saved_count}"
    )

    total_files = len(
        [
            file
            for file in os.listdir(
                OUTPUT_FOLDER
            )

            if file.lower().endswith(
                ".wav"
            )
        ]
    )

    print(
        f"Total live fake samples: {total_files}"
    )


if __name__ == "__main__":

    collect_live_fake()