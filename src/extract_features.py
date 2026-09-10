import numpy as np
import librosa
from pathlib import Path

SR = 16000
FRAME_MS = 20
HOP_MS = 10
N_BANDS = 16          # number of frequency bands per frame — small on purpose,
                        # this becomes your model's input size (embedded-budget friendly)

FRAME_LEN = int(SR * FRAME_MS / 1000)
HOP_LEN = int(SR * HOP_MS / 1000)

def extract_band_energies(filepath, label):
    y, _ = librosa.load(filepath, sr=SR)

    # STFT: converts audio into time-frequency representation
    stft = np.abs(librosa.stft(y, n_fft=512, hop_length=HOP_LEN, win_length=FRAME_LEN))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)

    # Split spectrum into N_BANDS log-spaced bands (mimics Bark-scale grouping —
    # low frequencies get finer resolution, matching human hearing sensitivity)
    band_edges = np.logspace(np.log10(50), np.log10(SR/2), N_BANDS + 1)

    features = []
    for t in range(stft.shape[1]):
        frame_energies = []
        for b in range(N_BANDS):
            mask = (freqs >= band_edges[b]) & (freqs < band_edges[b+1])
            energy = np.sum(stft[mask, t] ** 2) if mask.any() else 0.0
            frame_energies.append(energy)
        features.append(frame_energies)

    features = np.array(features)
    features = np.log1p(features)   # log-compress: energy varies over huge ranges,
                                      # log scale makes it learnable (same reason dB scales exist)

    labels = np.full(len(features), label)
    return features, labels

if __name__ == "__main__":
    # label mapping: 0=speech, 1=stationary, 2=non-stationary, 3=impulsive
    files_labels = [
        ("data/raw/clean_speech_01.flac", 0),
        ("data/raw/stationary_noise.wav", 1),
        ("data/raw/nonstationary_wind_sound.wav", 2),
        ("data/raw/battlefield-7.flac", 3),
    ]

    all_features, all_labels = [], []
    for filepath, label in files_labels:
        feats, labs = extract_band_energies(filepath, label)
        all_features.append(feats)
        all_labels.append(labs)
        print(f"{filepath}: {feats.shape[0]} frames extracted, feature dim={feats.shape[1]}")

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    Path("data/processed").mkdir(exist_ok=True, parents=True)
    np.save("data/processed/X_features.npy", X)
    np.save("data/processed/y_labels.npy", y)
    print(f"\nSaved dataset: X shape={X.shape}, y shape={y.shape}")