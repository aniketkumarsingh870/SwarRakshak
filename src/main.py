from audio_recorder import record_audio
from audio_processor import load_audio
from feature_extractor import extract_mfcc, prepare_features
from detector import analyze_voice


def main():

    print("=" * 50)
    print("SWAR RAKSHAK")
    print("AI Voice Impersonation Detection System")
    print("=" * 50)

    # Audio file path
    audio_path = "../data/samples/test_audio.wav"

    # Step 1: Record Audio
    print("\nStep 1: Recording Audio...")
    record_audio(filename=audio_path)

    # Step 2: Load and Process Audio
    print("\nStep 2: Processing Audio...")
    audio_data, sample_rate = load_audio(audio_path)

    # Step 3: Extract MFCC Features
    print("\nStep 3: Extracting Voice Features...")
    mfcc = extract_mfcc(audio_data, sample_rate)

    # Step 4: Prepare Features
    print("\nStep 4: Preparing Features...")
    features = prepare_features(mfcc)

    # Step 5: Analyze Voice
    print("\nStep 5: Analyzing Voice...")
    risk_score, risk_level = analyze_voice(features)

    # Final Result
    print("\n" + "=" * 50)
    print("VOICE ANALYSIS RESULT")
    print("=" * 50)

    print(f"Impersonation Risk Score: {risk_score:.2f}%")
    print(f"Risk Level: {risk_level}")

    print("=" * 50)


if __name__ == "__main__":
    main()