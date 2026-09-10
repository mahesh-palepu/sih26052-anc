import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SR = 16000  # 16kHz: standard for speech/embedded audio work (not 44.1kHz — 
            # we don't need music-grade fidelity, and lower SR = less compute 
            # on the MCU later)

def load_and_plot(filepath, title):
    y, sr = librosa.load(filepath, sr=SR)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # Time domain
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title(f"{title} — Waveform")
    
    # Frequency domain (spectrogram)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set_title(f"{title} — Spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    
    plt.tight_layout()
    plt.savefig(f"data/processed/{title}_analysis.png")
    plt.show()
    
    return y, sr

if __name__ == "__main__":
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Run this once per file you downloaded — swap in your actual filenames
    load_and_plot("data/raw/helicopter-engine-noise.wav", "stationary_engine")
    