import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
from pystoi import stoi

SR = 16000
N_BANDS = 16
FRAME_LEN = 320
HOP_LEN = 160
SEQ_LEN = 32

class SuppressionNet(nn.Module):
    def __init__(self, in_dim=16, hidden=24, out_dim=16):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, out_dim), nn.Sigmoid())

    def forward(self, x):
        h, _ = self.gru(x)
        return self.out(h)

def get_band_masks(freqs, n_bands=N_BANDS):
    band_edges = np.logspace(np.log10(50), np.log10(SR/2), n_bands + 1)
    masks = []
    for b in range(n_bands):
        masks.append((freqs >= band_edges[b]) & (freqs < band_edges[b+1]))
    return masks

def suppress_audio(model, noisy_path, out_path):
    y, _ = librosa.load(noisy_path, sr=SR)
    stft = librosa.stft(y, n_fft=512, hop_length=HOP_LEN, win_length=FRAME_LEN)
    mag, phase = np.abs(stft), np.angle(stft)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    band_masks = get_band_masks(freqs)

    # band energy -> log input feature (same as training)
    band_energy = np.zeros((mag.shape[1], N_BANDS))
    for b, mask in enumerate(band_masks):
        band_energy[:, b] = np.sum(mag[mask, :] ** 2, axis=0) if mask.any() else 0.0
    input_feat = np.log1p(band_energy).astype(np.float32)

    # pad to multiple of SEQ_LEN so the GRU can process it in chunks
    n_frames = input_feat.shape[0]
    pad = (SEQ_LEN - n_frames % SEQ_LEN) % SEQ_LEN
    input_padded = np.pad(input_feat, ((0, pad), (0, 0)))
    input_seq = input_padded.reshape(-1, SEQ_LEN, N_BANDS)

    with torch.no_grad():
        gains = model(torch.from_numpy(input_seq)).numpy()
    gains = gains.reshape(-1, N_BANDS)[:n_frames]   # remove padding

    # apply per-band gain to the magnitude spectrum, per frequency bin
    mag_gained = mag.copy()
    for b, mask in enumerate(band_masks):
        if mask.any():
            mag_gained[mask, :] *= gains[:, b]

    # reconstruct waveform: gained magnitude + original phase, inverse STFT
    stft_gained = mag_gained * np.exp(1j * phase)
    y_clean = librosa.istft(stft_gained, hop_length=HOP_LEN, win_length=FRAME_LEN)

    sf.write(out_path, y_clean, SR)
    return y_clean

if __name__ == "__main__":
    model = SuppressionNet()
    model.load_state_dict(torch.load("data/processed/suppression_model.pt"))
    model.eval()

    # Test on battlefield noise mixed with speech at 0dB (a hard, realistic case)
    clean, _ = librosa.load("data/raw/clean_speech_01.flac", sr=SR)
    noise, _ = librosa.load("data/raw/battlefield-7.flac", sr=SR)
    n = min(len(clean), len(noise))
    clean, noise = clean[:n], noise[:n]

    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    scale = np.sqrt(clean_power / (10 ** (0 / 10)) / (noise_power + 1e-10))
    noisy = clean + scale * noise
    sf.write("data/processed/test_noisy_battlefield.wav", noisy, SR)

    cleaned = suppress_audio(model, "data/processed/test_noisy_battlefield.wav",
                              "data/processed/test_cleaned_battlefield.wav")

    n2 = min(len(clean), len(noisy))
    before = stoi(clean[:n2], noisy[:n2], SR, extended=False)
    n3 = min(len(clean), len(cleaned))
    after = stoi(clean[:n3], cleaned[:n3], SR, extended=False)

    print(f"Battlefield noise @ 0dB SNR")
    print(f"STOI before (noisy):        {before:.3f}")
    print(f"STOI after (AI suppression): {after:.3f}")
    print(f"Improvement: {after - before:+.3f}")
    print(f"\n(Compare this to NLMS on similar impulsive noise: it only reached ~0.528)")