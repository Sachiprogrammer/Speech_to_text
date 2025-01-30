

!pip install openai-whisper ffmpeg-python jiwer pandas

import whisper
from jiwer import wer, cer
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def transcribe_audio(audio_path, model_size="medium"):
    """Transcribes an audio file using Whisper."""
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"], result["segments"]

def evaluate_transcription(ground_truth, transcribed_text):
    """Compares transcribed text with ground truth using WER and CER."""
    wer_score = wer(ground_truth, transcribed_text)
    cer_score = cer(ground_truth, transcribed_text)
    return wer_score, cer_score

def plot_waveform(audio_path):
    """Plots the waveform of the audio file."""
    sample_rate, audio_data = wavfile.read(audio_path)
    plt.figure(figsize=(12, 4))
    plt.plot(np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data)), audio_data)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

def plot_spectrogram(audio_path):
    """Plots the spectrogram of the audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    spectrogram = np.abs(librosa.stft(y))
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    plt.show()

# Example usage
audio_path = "/content/si1279.wav"  # Replace with actual audio file
transcribed_text, segments = transcribe_audio(audio_path)

# Ground Truth (Replace with actual transcript)
ground_truth = "This is the correct transcript of the audio."

# Evaluate Accuracy
wer_score, cer_score = evaluate_transcription(ground_truth, transcribed_text)

print("Transcribed Text:", transcribed_text)
print(f"Word Error Rate (WER): {wer_score:.2%}")
print(f"Character Error Rate (CER): {cer_score:.2%}")

# Plot audio waveform and spectrogram
plot_waveform(audio_path)
plot_spectrogram(audio_path)

