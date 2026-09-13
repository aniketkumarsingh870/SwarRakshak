import sounddevice as sd
import soundfile as sf


def record_audio(duration=5, sample_rate=16000, filename="recorded_audio.wav"):
    """
    Records audio from the microphone and saves it as a WAV file.
    """

    print("\nRecording started...")
    print(f"Recording for {duration} seconds. Please speak into the microphone.")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )

    # Wait until recording is complete
    sd.wait()

    print("Recording finished.")

    # Save the audio
    sf.write(filename, audio, sample_rate)

    print(f"Audio successfully saved at: {filename}")