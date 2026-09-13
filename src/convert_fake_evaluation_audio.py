import os
from pydub import AudioSegment


INPUT_FOLDER = "data/evaluation/fake_original"
OUTPUT_FOLDER = "data/evaluation/fake"

SAMPLE_RATE = 16000

SUPPORTED_FORMATS = (
    ".mp3",
    ".mp4",
    ".mpeg",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".wav",
    ".flac"
)


os.makedirs(OUTPUT_FOLDER, exist_ok=True)


files = [
    file
    for file in os.listdir(INPUT_FOLDER)
    if file.lower().endswith(SUPPORTED_FORMATS)
]


print("=" * 60)
print("SWARRAKSHAK FAKE EVALUATION AUDIO CONVERTER")
print("=" * 60)

print(f"\nFiles found: {len(files)}\n")


converted = 0


for file_name in files:

    input_path = os.path.join(
        INPUT_FOLDER,
        file_name
    )

    try:

        print(f"Converting: {file_name}")

        audio = AudioSegment.from_file(
            input_path
        )

        # Convert to mono
        audio = audio.set_channels(1)

        # Convert to 16 kHz
        audio = audio.set_frame_rate(
            SAMPLE_RATE
        )

        converted += 1

        output_path = os.path.join(
            OUTPUT_FOLDER,
            f"fake_eval_{converted:03d}.wav"
        )

        audio.export(
            output_path,
            format="wav"
        )

        print(f"Saved -> {output_path}")

    except Exception as error:

        print(f"\nERROR converting: {file_name}")
        print(error)


print("\n" + "=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print(f"\nSuccessfully converted: {converted}")
print(f"Output folder: {OUTPUT_FOLDER}")
print("Format: WAV | 16000 Hz | Mono")