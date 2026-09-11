from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from pystoi import stoi


SR = 16000
N_BANDS = 16
N_FFT = 512
FRAME_LEN = 320
HOP_LEN = 160
SEQ_LEN = 32
SMOOTH_ALPHA = 0.60


class SuppressionNet(nn.Module):
    def __init__(self, in_dim=N_BANDS, hidden=64, out_dim=N_BANDS):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        h, _ = self.gru(x)
        return self.out(h)


def get_band_masks(freqs, n_bands=N_BANDS):
    band_edges = np.logspace(
        np.log10(50),
        np.log10(SR / 2),
        n_bands + 1
    )

    masks = []
    for b in range(n_bands):
        mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        masks.append(mask)

    return masks


def smooth_gains(gains, alpha=SMOOTH_ALPHA):
    """
    Apply causal exponential smoothing over time frames.

    alpha = 0.0: no smoothing
    alpha closer to 1.0: smoother but slower response to sudden changes
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in the range [0.0, 1.0).")

    if len(gains) == 0:
        return gains

    smoothed = np.empty_like(gains)
    smoothed[0] = gains[0]

    for t in range(1, len(gains)):
        smoothed[t] = alpha * smoothed[t - 1] + (1.0 - alpha) * gains[t]

    return np.clip(smoothed, 0.0, 1.0)


def get_input_features(mag, band_masks):
    """
    Convert STFT magnitudes into frame-wise, 16-band log-energy features.
    Output shape: (n_frames, N_BANDS)
    """
    band_energy = np.zeros(
        (mag.shape[1], N_BANDS),
        dtype=np.float32
    )

    for b, mask in enumerate(band_masks):
        if mask.any():
            band_energy[:, b] = np.sum(mag[mask, :] ** 2, axis=0)

    return np.log1p(band_energy).astype(np.float32)


def predict_gains(model, input_feat):
    """
    Predict a gain vector per time frame.
    Pads frames to a multiple of SEQ_LEN, then removes padding afterward.
    """
    n_frames = input_feat.shape[0]

    if n_frames == 0:
        return np.empty((0, N_BANDS), dtype=np.float32)

    pad = (SEQ_LEN - n_frames % SEQ_LEN) % SEQ_LEN
    input_padded = np.pad(
        input_feat,
        ((0, pad), (0, 0)),
        mode="constant"
    )

    input_seq = input_padded.reshape(-1, SEQ_LEN, N_BANDS)
    input_tensor = torch.from_numpy(input_seq)

    with torch.no_grad():
        gains = model(input_tensor).cpu().numpy()

    gains = gains.reshape(-1, N_BANDS)[:n_frames]
    return np.clip(gains, 0.0, 1.0)


def suppress_audio(model, noisy_path, out_path, smooth_alpha=SMOOTH_ALPHA):
    noisy_path = Path(noisy_path)
    out_path = Path(out_path)

    if not noisy_path.exists():
        raise FileNotFoundError(f"Noisy input file not found: {noisy_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    y, _ = librosa.load(noisy_path, sr=SR, mono=True)

    if len(y) == 0:
        raise ValueError(f"Input audio is empty: {noisy_path}")

    stft = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        win_length=FRAME_LEN
    )

    mag = np.abs(stft)
    phase = np.angle(stft)

    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    band_masks = get_band_masks(freqs)

    input_feat = get_input_features(mag, band_masks)
    gains = predict_gains(model, input_feat)

    # Secondary temporal smoothing filter:
    # reduces frame-to-frame gain fluctuations and musical-noise artifacts.
    gains = smooth_gains(gains, alpha=smooth_alpha)

    mag_gained = mag.copy()

    for b, mask in enumerate(band_masks):
        if mask.any():
            mag_gained[mask, :] *= gains[:, b][None, :]

    stft_gained = mag_gained * np.exp(1j * phase)

    y_clean = librosa.istft(
        stft_gained,
        hop_length=HOP_LEN,
        win_length=FRAME_LEN,
        length=len(y)
    )

    y_clean = np.clip(y_clean, -1.0, 1.0)
    sf.write(out_path, y_clean, SR)

    return y_clean


def mix_at_snr(clean, noise, snr_db):
    n = min(len(clean), len(noise))

    if n == 0:
        raise ValueError("Clean speech or noise signal is empty.")

    clean = clean[:n]
    noise = noise[:n]

    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10

    scale = np.sqrt(
        clean_power / (10 ** (snr_db / 10)) / noise_power
    )

    noisy = clean + scale * noise
    return clean, noisy


if __name__ == "__main__":
    model_path = Path("data/processed/suppression_model.pt")
    clean_path = Path("data/raw/clean_speech_01.flac")
    noise_path = Path("data/raw/impulsive_battlefield-7.flac")

    noisy_out_path = Path("data/processed/test_noisy_battlefield.wav")

    for path in [model_path, clean_path, noise_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    model = SuppressionNet(hidden=64)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    clean, _ = librosa.load(clean_path, sr=SR, mono=True)
    noise, _ = librosa.load(noise_path, sr=SR, mono=True)

    clean, noisy = mix_at_snr(clean, noise, snr_db=0.0)

    noisy_out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(noisy_out_path, noisy, SR)

    # Baseline STOI on the noisy mixture
    n_before = min(len(clean), len(noisy))
    before = stoi(clean[:n_before], noisy[:n_before], SR, extended=False)

    print("Battlefield noise @ 0 dB SNR\n")
    print(f"STOI before (noisy): {before:.3f}\n")

    # Sweep over smoothing alphas
    alpha_grid = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]

    results = []

    for alpha in alpha_grid:
        cleaned_path = Path(f"data/processed/test_cleaned_alpha{alpha:.1f}.wav")

        cleaned = suppress_audio(
            model=model,
            noisy_path=noisy_out_path,
            out_path=cleaned_path,
            smooth_alpha=alpha
        )

        n_after = min(len(clean), len(cleaned))
        after = stoi(clean[:n_after], cleaned[:n_after], SR, extended=False)
        gain = after - before

        results.append((alpha, after, gain))

    # Print a compact table
    print(f"{'alpha':>5} | {'STOI after':>10} | {'gain':>8}")
    print("-" * 28)
    for alpha, after, gain in results:
        print(f"{alpha:5.1f} | {after:10.3f} | {gain:+8.3f}")

    print(
        "\nTip: listen to data/processed/test_cleaned_alpha*.wav "
        "to choose the best alpha by ear, not just by STOI."
    )