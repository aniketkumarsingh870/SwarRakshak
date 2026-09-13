import os
import sounddevice as sd
import soundfile as sf
import numpy as np


# ==========================================
# SWARRAKSHAK
# LIVE GENUINE VOICE DATA COLLECTOR
# ==========================================


SAMPLE_RATE = 16000
DURATION = 5

OUTPUT_FOLDER = "data/live_real"

# Silence threshold
MIN_RMS = 0.005


def calculate_rms(audio):
    return np.sqrt(
        np.mean(
            np.square(audio)
        )
    )


def get_next_index():

    if not os.path.exists(OUTPUT_FOLDER):
        return 0

    files = [
        file
        for file in os.listdir(OUTPUT_FOLDER)
        if file.endswith(".wav")
    ]

    indices = []

    for file in files:

        try:
            number = int(
                file
                .replace("live_real_", "")
                .replace(".wav", "")
            )

            indices.append(number)

        except ValueError:
            pass

    if not indices:
        return 0

    return max(indices) + 1


def collect_live_real():

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    print("\n")
    print("=" * 65)
    print("SWARRAKSHAK LIVE REAL VOICE COLLECTION")
    print("=" * 65)

    print("\nRecording Settings")
    print(f"Sample Rate : {SAMPLE_RATE} Hz")
    print(f"Duration    : {DURATION} seconds")

    print("\nImportant:")
    print("- Speak naturally.")
    print("- Use different sentences.")
    print("- Do not imitate an AI voice.")
    print("- Keep normal background conditions.")
    print("- Some recordings can be closer/farther from the microphone.")

    try:

        total_samples = int(
            input(
                "\nHow many genuine samples do you want to record? "
            )
        )

    except ValueError:

        print("\nInvalid number.")
        return

    if total_samples <= 0:

        print("\nNumber of samples must be greater than 0.")
        return

    current_index = get_next_index()

    saved_count = 0

    while saved_count < total_samples:

        print("\n" + "-" * 65)

        print(
            f"Sample {saved_count + 1}/{total_samples}"
        )

        input(
            "Press ENTER when ready..."
        )

        print(
            f"Recording for {DURATION} seconds..."
        )

        audio = sd.rec(
            int(
                SAMPLE_RATE * DURATION
            ),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        audio = audio.flatten()

        rms = calculate_rms(audio)

        print(
            f"Audio RMS: {rms:.6f}"
        )

        if rms < MIN_RMS:

            print(
                "Recording too quiet. Sample not saved."
            )

            print(
                "Please speak a little louder and try again."
            )

            continue

        filename = os.path.join(
            OUTPUT_FOLDER,
            f"live_real_{current_index:04d}.wav"
        )

        sf.write(
            filename,
            audio,
            SAMPLE_RATE
        )

        print(
            f"Saved: {filename}"
        )

        saved_count += 1
        current_index += 1

    print("\n")
    print("=" * 65)
    print("LIVE REAL DATA COLLECTION COMPLETE")
    print("=" * 65)

    print(
        f"\nNew samples recorded: {saved_count}"
    )

    total_files = len(
        [
            file
            for file in os.listdir(OUTPUT_FOLDER)
            if file.endswith(".wav")
        ]
    )

    print(
        f"Total live real samples available: {total_files}"
    )


if __name__ == "__main__":

    collect_live_real()