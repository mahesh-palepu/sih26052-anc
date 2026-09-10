import numpy as np
import librosa
import soundfile as sf
from pystoi import stoi

SR = 16000

def nlms_filter(reference, primary, filter_len=128, mu=0.5, eps=1e-6):
    """
    reference: signal correlated with the noise (your 'engine_noise.wav' alone)
    primary:   signal to be cleaned (speech + noise mix)
    filter_len: number of taps (filter 'memory' length) — longer = more noise
                structure captured, but slower to converge & more compute
                (this directly maps to RAM/CPU budget on your future MCU)
    mu: step size — higher = faster adaptation, but risk of instability
    """
    n = min(len(reference), len(primary))
    reference, primary = reference[:n], primary[:n]

    w = np.zeros(filter_len)          # adaptive filter coefficients (start at zero)
    output = np.zeros(n)              # cleaned signal
    x_buf = np.zeros(filter_len)      # sliding window of reference signal

    for i in range(n):
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = reference[i]

        y = np.dot(w, x_buf)          # filter's current noise estimate
        e = primary[i] - y            # error = cleaned output (this is your output!)
        output[i] = e

        # NLMS weight update — normalized by input power (+eps avoids div-by-zero)
        norm = np.dot(x_buf, x_buf) + eps
        w += (mu * e / norm) * x_buf

    return output

if __name__ == "__main__":
    # Load your reference noise-only clip and the noisy speech mix
    noise_ref, _ = librosa.load("data/raw/battlefield-7.flac", sr=SR)
    noisy_mix, _ = librosa.load("data/raw/speech_battlefield.wav.", sr=SR)
    clean_speech, _ = librosa.load("data/raw/clean_speech_01.flac", sr=SR)

    cleaned = nlms_filter(noise_ref, noisy_mix, filter_len=128, mu=0.5)

    sf.write("data/processed/nlms_cleaned.wav", cleaned, SR)

    # Compare STOI: before vs after
    n = min(len(clean_speech), len(noisy_mix))
    before_score = stoi(clean_speech[:n], noisy_mix[:n], SR, extended=False)

    n2 = min(len(clean_speech), len(cleaned))
    after_score = stoi(clean_speech[:n2], cleaned[:n2], SR, extended=False)

    print(f"STOI before NLMS: {before_score:.3f}")
    print(f"STOI after  NLMS: {after_score:.3f}")
    print(f"Improvement: {after_score - before_score:+.3f}")