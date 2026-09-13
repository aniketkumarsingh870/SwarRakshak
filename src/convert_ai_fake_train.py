import os
from pydub import AudioSegment


INPUT_FOLDER = "data/ai_fake_train"
OUTPUT_FOLDER = "data/ai_fake_train_wav"

SAMPLE_RATE = 16000
CLIP_DURATION_MS = 4000   # 4 seconds

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


def split_fake_audio():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Remove previously generated clips
    for old_file in os.listdir(OUTPUT_FOLDER):
        if old_file.lower().endswith(".wav"):
            os.remove(
                os.path.join(
                    OUTPUT_FOLDER,
                    old_file
                )
            )

    files = [
        file
        for file in os.listdir(INPUT_FOLDER)
        if file.lower().endswith(SUPPORTED_FORMATS)
    ]

    print("\n" + "=" * 65)
    print("SWARRAKSHAK AI FAKE AUDIO SPLITTER")
    print("=" * 65)

    print(f"\nSource files found: {len(files)}")

    clip_count = 0

    for file_name in files:

        input_path = os.path.join(
            INPUT_FOLDER,
            file_name
        )

        print(
            f"\nProcessing: {file_name}"
        )

        try:

            audio = AudioSegment.from_file(
                input_path
            )

            # Mono
            audio = audio.set_channels(1)

            # 16 kHz
            audio = audio.set_frame_rate(
                SAMPLE_RATE
            )

            duration_seconds = (
                len(audio) / 1000
            )

            print(
                f"Duration: {duration_seconds:.2f} seconds"
            )

            # ----------------------------------
            # Split into 4-second chunks
            # ----------------------------------

            for start_ms in range(
                0,
                len(audio),
                CLIP_DURATION_MS
            ):

                end_ms = (
                    start_ms
                    +
                    CLIP_DURATION_MS
                )

                clip = audio[
                    start_ms:end_ms
                ]

                # Ignore very short final pieces
                if len(clip) < 2000:
                    continue

                clip_count += 1

                output_name = (
                    f"ai_fake_train_"
                    f"{clip_count:04d}.wav"
                )

                output_path = os.path.join(
                    OUTPUT_FOLDER,
                    output_name
                )

                clip.export(
                    output_path,
                    format="wav"
                )

                print(
                    f"Created: {output_name}"
                )

        except Exception as error:

            print(
                f"ERROR: {file_name}"
            )

            print(error)

    print("\n" + "=" * 65)
    print("SPLITTING COMPLETE")
    print("=" * 65)

    print(
        f"\nTotal fake training clips: {clip_count}"
    )

    print(
        f"Saved in: {OUTPUT_FOLDER}"
    )

    print(
        "Format: WAV | 16000 Hz | Mono"
    )


if __name__ == "__main__":

    split_fake_audio()