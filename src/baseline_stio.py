from pystoi import stoi
import librosa

SR = 16000

clean, _ = librosa.load("data/raw/clean_speech_01.flac", sr=SR)
noisy, _ = librosa.load("data/raw/Speech_plus_engine.wav", sr=SR)  

# Trim to same length
min_len = min(len(clean), len(noisy))
clean, noisy = clean[:min_len], noisy[:min_len]

score = stoi(clean, noisy, SR, extended=False)
print(f"Baseline STOI (no denoising): {score:.3f}")
# STOI ranges 0-1. Below ~0.5 = poor intelligibility. 
# This is your "before" number — Step 2's NLMS filter should push it up.