import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from pystoi import stoi

SR = 16000
N_BANDS = 16
N_FFT = 512
FRAME_LEN = 320
HOP_LEN = 160

def get_band_masks(freqs, n_bands=N_BANDS):
    band_edges = np.logspace(np.log10(50), np.log10(SR/2), n_bands + 1)
    return [(freqs >= band_edges[b]) & (freqs < band_edges[b+1]) for b in range(n_bands)]

interpreter = tf.lite.Interpreter(model_path="data/processed/suppression_model_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
feat_in = next(d for d in input_details if d['shape'][-1] == 16)
hid_in  = next(d for d in input_details if d['shape'][-1] == 64)
gain_out = next(d for d in output_details if d['shape'][-1] == 16)
hid_out  = next(d for d in output_details if d['shape'][-1] == 64)

# Same test case as your PyTorch pipeline: battlefield noise @ 0dB SNR
clean, _ = librosa.load("data/raw/clean_speech_01.flac", sr=SR)
noise, _ = librosa.load("data/raw/impulsive_battlefield-7.flac", sr=SR)
n = min(len(clean), len(noise))
clean, noise = clean[:n], noise[:n]
clean_power, noise_power = np.mean(clean**2), np.mean(noise**2)
scale = np.sqrt(clean_power / (10**(0/10)) / (noise_power + 1e-10))
noisy = clean + scale * noise

stft = librosa.stft(noisy, n_fft=N_FFT, hop_length=HOP_LEN, win_length=FRAME_LEN)
mag, phase = np.abs(stft), np.angle(stft)
freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
band_masks = get_band_masks(freqs)

band_energy = np.zeros((mag.shape[1], N_BANDS), dtype=np.float32)
for b, mask in enumerate(band_masks):
    if mask.any():
        band_energy[:, b] = np.sum(mag[mask, :] ** 2, axis=0)
input_feat = np.log1p(band_energy).astype(np.float32)

# Stream frame-by-frame through TFLite, exactly like real embedded inference
f_scale, f_zp = feat_in['quantization']
h_scale, h_zp = hid_in['quantization']
g_scale, g_zp = gain_out['quantization']
ho_scale, ho_zp = hid_out['quantization']

hidden = np.zeros((1, 64), dtype=np.float32)
all_gains = []
for t in range(input_feat.shape[0]):
    frame = input_feat[t:t+1]
    frame_q = np.clip(np.round(frame / f_scale + f_zp), -128, 127).astype(np.int8)
    hidden_q = np.clip(np.round(hidden / h_scale + h_zp), -128, 127).astype(np.int8)

    interpreter.set_tensor(feat_in['index'], frame_q)
    interpreter.set_tensor(hid_in['index'], hidden_q)
    interpreter.invoke()

    gain_q = interpreter.get_tensor(gain_out['index'])
    hnew_q = interpreter.get_tensor(hid_out['index'])

    gain_deq = (gain_q.astype(np.float32) - g_zp) * g_scale
    hidden = (hnew_q.astype(np.float32) - ho_zp) * ho_scale

    all_gains.append(gain_deq.flatten())

gains = np.clip(np.array(all_gains), 0.0, 1.0)

mag_gained = mag.copy()
for b, mask in enumerate(band_masks):
    if mask.any():
        mag_gained[mask, :] *= gains[:, b][None, :]

stft_gained = mag_gained * np.exp(1j * phase)
y_clean = librosa.istft(stft_gained, hop_length=HOP_LEN, win_length=FRAME_LEN, length=len(noisy))
y_clean = np.clip(y_clean, -1.0, 1.0)
sf.write("data/processed/test_cleaned_tflite.wav", y_clean, SR)

n2 = min(len(clean), len(noisy))
before = stoi(clean[:n2], noisy[:n2], SR, extended=False)
n3 = min(len(clean), len(y_clean))
after = stoi(clean[:n3], y_clean[:n3], SR, extended=False)

print(f"Battlefield noise @ 0dB SNR — Full-clip TFLite int8 streaming inference")
print(f"STOI before: {before:.3f}")
print(f"STOI after (TFLite int8):  {after:.3f}")
print(f"Improvement: {after - before:+.3f}")
print(f"\n(Compare to PyTorch pipeline: +0.039 raw / +0.033 with alpha=0.3 smoothing)")