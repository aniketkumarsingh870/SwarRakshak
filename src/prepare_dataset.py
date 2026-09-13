import os
import pandas as pd

from feature_extractor import (
    extract_features,
    get_feature_names
)


# ==========================================
# TRAINING DATA ONLY
# ==========================================

DATA_SOURCES = [

    # Original REAL benchmark
    {
        "folder": "data/dataset/real",
        "label": 0,
        "source": "benchmark_real"
    },

    # Your microphone REAL recordings
    {
        "folder": "data/live_real",
        "label": 0,
        "source": "live_real"
    },

    # Original FAKE benchmark
    {
        "folder": "data/dataset/fake",
        "label": 1,
        "source": "benchmark_fake"
    },

    # Microphone/replayed fake recordings
    {
        "folder": "data/live_fake",
        "label": 1,
        "source": "live_fake"
    },

    # NEW modern AI-generated fake voices
    {
        "folder": "data/ai_fake_train_wav",
        "label": 1,
        "source": "modern_ai_fake"
    }
]


OUTPUT_FILE = "data/processed/voice_features.csv"


def prepare_dataset():

    print("\n" + "=" * 70)
    print("SWARRAKSHAK TRAINING DATASET PREPARATION")
    print("=" * 70)

    feature_names = get_feature_names()

    print(
        f"\nFeatures per audio: {len(feature_names)}"
    )

    rows = []

    total_processed = 0
    total_failed = 0


    # ======================================
    # PROCESS DATA SOURCES
    # ======================================

    for source_info in DATA_SOURCES:

        folder = source_info["folder"]
        label = source_info["label"]
        source_name = source_info["source"]

        print("\n" + "-" * 70)

        print(
            f"Source : {source_name}"
        )

        print(
            f"Folder : {folder}"
        )

        print(
            f"Label  : {'REAL' if label == 0 else 'FAKE'}"
        )

        print("-" * 70)


        if not os.path.exists(folder):

            print(
                "Folder does not exist - skipping."
            )

            continue


        files = [
            file
            for file in os.listdir(folder)
            if file.lower().endswith(".wav")
        ]


        print(
            f"Files found: {len(files)}"
        )


        source_processed = 0


        for index, file_name in enumerate(
            files,
            start=1
        ):

            file_path = os.path.join(
                folder,
                file_name
            )


            try:

                features = extract_features(
                    file_path
                )


                if len(features) != len(feature_names):

                    print(
                        f"Feature mismatch: {file_name}"
                    )

                    total_failed += 1

                    continue


                row = dict(
                    zip(
                        feature_names,
                        features
                    )
                )


                row["label"] = label

                row["source"] = source_name

                row["filename"] = file_name


                rows.append(
                    row
                )


                source_processed += 1

                total_processed += 1


                if (
                    index % 25 == 0
                    or
                    index == len(files)
                ):

                    print(
                        f"Processed {index}/{len(files)}"
                    )


            except Exception as error:

                total_failed += 1

                print(
                    f"\nERROR processing {file_name}"
                )

                print(error)


        print(
            f"Successfully processed: {source_processed}"
        )


    # ======================================
    # CHECK
    # ======================================

    if len(rows) == 0:

        print(
            "\nERROR: No training samples processed."
        )

        return


    # ======================================
    # CREATE DATAFRAME
    # ======================================

    df = pd.DataFrame(
        rows
    )


    # Shuffle training dataset
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(
        drop=True
    )


    # ======================================
    # SAVE
    # ======================================

    os.makedirs(
        "data/processed",
        exist_ok=True
    )


    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ======================================
    # REPORT
    # ======================================

    print("\n" + "=" * 70)
    print("TRAINING DATASET READY")
    print("=" * 70)


    print(
        f"\nTotal samples: {len(df)}"
    )

    print(
        f"Successfully processed: {total_processed}"
    )

    print(
        f"Failed: {total_failed}"
    )

    print(
        f"Features per sample: {len(feature_names)}"
    )


    print(
        "\nClass distribution:"
    )

    print(
        df["label"].value_counts()
    )


    print(
        "\nSource distribution:"
    )

    print(
        df["source"].value_counts()
    )


    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


    print("\nIMPORTANT:")

    print(
        "data/evaluation/real was NOT used."
    )

    print(
        "data/evaluation/fake was NOT used."
    )


if __name__ == "__main__":

    prepare_dataset()