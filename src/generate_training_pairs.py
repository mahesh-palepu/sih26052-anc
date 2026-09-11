import numpy as np
import librosa
from pathlib import Path

SR = 16000
N_BANDS = 16
FRAME_LEN = 320
HOP_LEN = 160


def get_band_energy_sequence(y):
    stft = librosa.stft(
        y,
        n_fft=512,
        hop_length=HOP_LEN,
        win_length=FRAME_LEN
    )
    mag = np.abs(stft)

    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    band_edges = np.logspace(
        np.log10(50),
        np.log10(SR / 2),
        N_BANDS + 1
    )

    band_energy = np.zeros((mag.shape[1], N_BANDS), dtype=np.float32)

    for b in range(N_BANDS):
        mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])

        if mask.any():
            band_energy[:, b] = np.sum(mag[mask, :] ** 2, axis=0)

    return band_energy, mag, freqs, band_edges


def mix_at_snr(clean, noise, snr_db):
    n = min(len(clean), len(noise))

    if n == 0:
        raise ValueError("Clean speech or noise audio is empty.")

    clean = clean[:n]
    noise = noise[:n]

    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10

    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)

    noisy = clean + scale * noise
    return clean, scale * noise, noisy


def build_pairs(clean_path, noise_paths, snr_levels=(-5, 0, 5, 10)):
    clean_path = Path(clean_path)

    if not clean_path.exists():
        raise FileNotFoundError(f"Clean speech file not found: {clean_path}")

    clean_full, _ = librosa.load(clean_path, sr=SR)

    if len(clean_full) == 0:
        raise ValueError(f"Clean speech file is empty: {clean_path}")

    all_X, all_Y = [], []

    for noise_path in noise_paths:
        noise_path = Path(noise_path)

        if not noise_path.exists():
            raise FileNotFoundError(f"Noise file not found: {noise_path}")

        noise_full, _ = librosa.load(noise_path, sr=SR)

        if len(noise_full) == 0:
            raise ValueError(f"Noise file is empty: {noise_path}")

        reps = int(np.ceil(len(clean_full) / len(noise_full)))
        noise_full = np.tile(noise_full, reps)[:len(clean_full)]

        for snr in snr_levels:
            clean_seg, _, noisy = mix_at_snr(clean_full, noise_full, snr)

            clean_be, _, _, _ = get_band_energy_sequence(clean_seg)
            noisy_be, _, _, _ = get_band_energy_sequence(noisy)

            n_frames = min(len(clean_be), len(noisy_be))
            clean_be = clean_be[:n_frames]
            noisy_be = noisy_be[:n_frames]

            gain_target = np.clip(
                np.sqrt(clean_be / (noisy_be + 1e-8)),
                0.0,
                1.0
            ).astype(np.float32)

            input_feat = np.log1p(noisy_be).astype(np.float32)

            all_X.append(input_feat)
            all_Y.append(gain_target)

            print(f"{noise_path.name} @ {snr:+} dB SNR: {n_frames} frames")

    return np.concatenate(all_X, axis=0), np.concatenate(all_Y, axis=0)


if __name__ == "__main__":
    noise_files = [
        "data/raw/stationary_noise.wav",
        "data/raw/stationary_floor_fan.wav",
        "data/raw/stationary_jet_plane_flying.wav",
        "data/raw/nonstationary_wind_sound.wav",
        "data/raw/nonstationary_rain_strong_winds.wav",
        "data/raw/impulsive_battlefield-7.flac",
        "data/raw/impulsive_warzone_01.wav",
    ]

    speech_files = [
        "data/raw/clean_speech_01.flac",
        "data/raw/clean_speech_02.flac",
        "data/raw/clean_speech_03.flac",
        "data/raw/clean_speech_04.flac",
        "data/raw/clean_speech_05.flac",
        "data/raw/clean_speech_06.flac",
        "data/raw/clean_speech_07.flac",
    ]

    all_X, all_Y = [], []

    for speech_path in speech_files:
        print(f"\nBuilding training pairs for: {speech_path}")
        X_part, Y_part = build_pairs(speech_path, noise_files)
        all_X.append(X_part)
        all_Y.append(Y_part)

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    Y = np.concatenate(all_Y, axis=0).astype(np.float32)

    Path("data/processed").mkdir(exist_ok=True, parents=True)

    np.save("data/processed/X_suppression.npy", X)
    np.save("data/processed/Y_suppression.npy", Y)

    print(f"\nFinal dataset: X={X.shape}, Y={Y.shape}")