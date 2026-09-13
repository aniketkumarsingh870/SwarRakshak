import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def load_audio(file_path):
    """
    Load an audio file.
    """

    audio, sample_rate = librosa.load(
        file_path,
        sr=None
    )

    print("\nAudio Information")
    print("-" * 30)
    print(f"Sample Rate: {sample_rate} Hz")

    duration = librosa.get_duration(
        y=audio,
        sr=sample_rate
    )

    print(f"Duration: {duration:.2f} seconds")

    return audio, sample_rate


def display_waveform(audio, sample_rate):
    """
    Display the waveform of the audio.
    """

    plt.figure(figsize=(10, 4))

    librosa.display.waveshow(
        audio,
        sr=sample_rate
    )

    plt.title("Audio Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.tight_layout()

    plt.show()


def display_spectrogram(audio, sample_rate):
    """
    Display the spectrogram of the audio.
    """

    plt.figure(figsize=(10, 4))

    spectrogram = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio)),
        ref=np.max
    )

    librosa.display.specshow(
        spectrogram,
        sr=sample_rate,
        x_axis="time",
        y_axis="hz"
    )

    plt.colorbar(format="%+2.0f dB")

    plt.title("Audio Spectrogram")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()

    plt.show()