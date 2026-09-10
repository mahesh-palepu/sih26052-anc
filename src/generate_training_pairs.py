import numpy as np
import librosa
from pathlib import Path

SR = 16000
N_BANDS = 16
FRAME_LEN = 320
HOP_LEN = 160

def get_band_energy_sequence(y):
    stft = librosa.stft(y, n_fft=512, hop_length=HOP_LEN, win_length=FRAME_LEN)
    mag = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    band_edges = np.logspace(np.log10(50), np.log10(SR/2), N_BANDS + 1)

    band_energy = np.zeros((mag.shape[1], N_BANDS))
    for b in range(N_BANDS):
        mask = (freqs >= band_edges[b]) & (freqs < band_edges[b+1])
        band_energy[:, b] = np.sum(mag[mask, :] ** 2, axis=0) if mask.any() else 0.0
    return band_energy, mag, freqs, band_edges

def mix_at_snr(clean, noise, snr_db):
    n = min(len(clean), len(noise))
    clean, noise = clean[:n], noise[:n]
    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    noisy = clean + scale * noise
    return clean, scale * noise, noisy

def build_pairs(clean_path, noise_paths, snr_levels=(-5, 0, 5, 10)):
    clean_full, _ = librosa.load(clean_path, sr=SR)
    all_X, all_Y = [], []

    for noise_path in noise_paths:
        noise_full, _ = librosa.load(noise_path, sr=SR)
        # loop noise if shorter than speech
        reps = int(np.ceil(len(clean_full) / len(noise_full)))
        noise_full = np.tile(noise_full, reps)[:len(clean_full)]

        for snr in snr_levels:
            clean_seg, noise_seg, noisy = mix_at_snr(clean_full, noise_full, snr)

            clean_be, _, _, _ = get_band_energy_sequence(clean_seg)
            noisy_be, _, _, _ = get_band_energy_sequence(noisy)

            n_frames = min(len(clean_be), len(noisy_be))
            clean_be, noisy_be = clean_be[:n_frames], noisy_be[:n_frames]

            # Ideal ratio mask: how much of the noisy energy is "real" (clean) signal
            gain_target = np.clip(np.sqrt(clean_be / (noisy_be + 1e-8)), 0.0, 1.0)

            input_feat = np.log1p(noisy_be)  # model sees log-compressed NOISY energy
            all_X.append(input_feat)
            all_Y.append(gain_target)

            print(f"{noise_path} @ {snr}dB SNR: {n_frames} frames")

    return np.concatenate(all_X, axis=0), np.concatenate(all_Y, axis=0)

if __name__ == "__main__":
    noise_files = [
        "data/raw/stationary_noise.wav",
        "data/raw/nonstationary_wind_sound.wav",
        "data/raw/battlefield-7.flac",
    ]

    X, Y = build_pairs("data/raw/clean_speech_01.flac", noise_files)

    Path("data/processed").mkdir(exist_ok=True, parents=True)
    np.save("data/processed/X_suppression.npy", X)
    np.save("data/processed/Y_suppression.npy", Y)
    print(f"\nFinal dataset: X={X.shape}, Y={Y.shape}")