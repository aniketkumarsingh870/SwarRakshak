from datasets import load_dataset, Audio
import os
import io
import soundfile as sf
from tqdm import tqdm


# ==========================================
# SWARRAKSHAK DATASET DOWNLOADER
# ASVspoof 2021 Logical Access Dataset
# ==========================================


DATASET_NAME = "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA"

REAL_FOLDER = "data/dataset/real"
FAKE_FOLDER = "data/dataset/fake"

MAX_REAL = 500
MAX_FAKE = 500


def create_folders():

    os.makedirs(REAL_FOLDER, exist_ok=True)
    os.makedirs(FAKE_FOLDER, exist_ok=True)


def download_dataset():

    print("=" * 60)
    print("SWARRAKSHAK DATASET DOWNLOADER")
    print("=" * 60)

    create_folders()

    print("\nLoading dataset...")
    print("Please wait...\n")

    dataset = load_dataset(
        DATASET_NAME,
        split="test",
        streaming=True
    )

    # Disable TorchCodec audio decoding
    dataset = dataset.cast_column(
        "audio",
        Audio(decode=False)
    )

    real_count = 0
    fake_count = 0

    print("Downloading audio samples...\n")

    for item in tqdm(dataset):

        label = item["label"]
        audio_data = item["audio"]

        # Get raw audio bytes
        audio_bytes = audio_data.get("bytes")

        if audio_bytes is None:
            continue

        try:

            audio_array, sample_rate = sf.read(
                io.BytesIO(audio_bytes)
            )

        except Exception:

            continue


        # --------------------------------
        # LABEL 0 = REAL / BONAFIDE
        # LABEL 1 = FAKE / SPOOF
        # --------------------------------

        if label == 0 and real_count < MAX_REAL:

            file_path = os.path.join(
                REAL_FOLDER,
                f"real_{real_count:04d}.wav"
            )

            sf.write(
                file_path,
                audio_array,
                sample_rate
            )

            real_count += 1


        elif label == 1 and fake_count < MAX_FAKE:

            file_path = os.path.join(
                FAKE_FOLDER,
                f"fake_{fake_count:04d}.wav"
            )

            sf.write(
                file_path,
                audio_array,
                sample_rate
            )

            fake_count += 1


        if real_count >= MAX_REAL and fake_count >= MAX_FAKE:
            break


    print("\n")
    print("=" * 60)
    print("DATASET DOWNLOAD COMPLETE")
    print("=" * 60)

    print(f"\nREAL samples: {real_count}")
    print(f"FAKE samples: {fake_count}")

    print("\nDataset saved successfully!")


if __name__ == "__main__":
    download_dataset()