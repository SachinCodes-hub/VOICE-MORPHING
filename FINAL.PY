#!/usr/bin/env python3
"""
Enhanced Voice Morphing Lab
- Records audio
- Shows DSP plots for each stage (raw, filtered, FFT, spectrogram, pitch)
- Applies morph: pitch shift + formant warp + additional effects
- Saves morphed file and plays back
"""

import os, time
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d
from tqdm import tqdm

# -----------------------
# Parameters
# -----------------------
FS = 44100
DURATION = 8            # seconds
OUTFILE = "morphed_voice.wav"
N_FFT = 2048

# -----------------------
# Utilities: filters, normalize
# -----------------------
def butter_bandpass(data, lowcut, highcut, fs, order=4):
    ny = 0.5 * fs
    b, a = butter(order, [lowcut/ny, min(highcut/ny, 0.9999)], btype='band')
    return lfilter(b, a, data)

def normalize(x, eps=1e-9):
    mx = np.max(np.abs(x)) + eps
    return x / mx

# -----------------------
# Formant warp function (frequency-domain spectral warping)
# -----------------------
def warp_formants(signal, fs, warp_factor=1.0, n_fft=N_FFT, hop_length=None):
    """
    Warp spectral magnitude by scaling frequency axis by warp_factor.
    warp_factor > 1.0 moves formants up (female-like), < 1.0 moves formants down.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    S = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)
    phase = np.angle(S)

    freqs = np.linspace(0, fs/2, mag.shape[0])
    # For each time frame, interpolate magnitude from freq/warp_factor -> freq
    warped_mag = np.zeros_like(mag)
    for t in range(mag.shape[1]):
        interp = interp1d(freqs, mag[:, t], bounds_error=False, fill_value=0.0)
        src_freqs = freqs / warp_factor
        warped_mag[:, t] = interp(src_freqs)

    # Reconstruct complex STFT with original phase (keeps phase coherence of pitched signal)
    S_warp = warped_mag * np.exp(1j * phase)
    y = librosa.istft(S_warp, hop_length=hop_length, length=len(signal))
    return y

# -----------------------
# Morphing modes using combination of techniques
# -----------------------
def morph_female(x, fs):
    # pitch up + small formant up warp
    y = librosa.effects.pitch_shift(x, sr=fs, n_steps=5, bins_per_octave=12)
    y = warp_formants(y, fs, warp_factor=1.12)
    # subtle harmonic excite
    y = np.tanh(1.1 * y)
    return normalize(y)

def morph_male(x, fs):
    # pitch down + formant down warp
    y = librosa.effects.pitch_shift(x, sr=fs, n_steps=-5, bins_per_octave=12)
    y = warp_formants(y, fs, warp_factor=0.88)
    # slight bass boost by amplifying low freq via filtering
    low = butter_bandpass(y, 50, 800, fs, order=3)
    y = normalize(0.9 * y + 0.2 * low)
    return y

def morph_robot(x, fs):
    # ring modulation + bit crush + spectral smoothing
    t = np.arange(len(x)) / fs
    carrier = np.sign(np.sin(2.0 * np.pi * 60.0 * t))  # square-ish carrier
    y = x * carrier
    # bit reduction
    y = np.round(y * 6.0) / 6.0
    # short echo
    delay = int(0.02 * fs)
    if delay < len(y):
        y[delay:] += 0.3 * y[:-delay]
    return normalize(y)

def morph_deep(x, fs):
    # deep voice: heavy pitch down + warp + saturation
    y = librosa.effects.pitch_shift(x, sr=fs, n_steps=-8)
    y = warp_formants(y, fs, warp_factor=0.85)
    y = np.tanh(1.5 * y)
    return normalize(y)

def morph_chipmunk(x, fs):
    # very high pitch and faster
    y = librosa.effects.pitch_shift(x, sr=fs, n_steps=8)
    y = librosa.effects.time_stretch(y, rate=1.25)
    y = warp_formants(y, fs, warp_factor=1.05)  # tiny formant shift
    return normalize(y)

# -----------------------
# Plot helpers (blocking)
# -----------------------
def plot_waveform(x, fs, title="Waveform"):
    plt.figure(figsize=(10, 3))
    times = np.arange(len(x))/fs
    plt.plot(times, x, color='dodgerblue')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()   # blocking until closed

def plot_spectrum(x, fs, title="Magnitude Spectrum"):
    X = np.fft.rfft(x * np.hanning(len(x)))
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    mags = np.abs(X)
    plt.figure(figsize=(10, 3))
    plt.semilogy(freqs, mags + 1e-8, color='magenta')
    plt.xlim(0, 8000)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (linear log-scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(x, fs, title="Spectrogram"):
    D = np.abs(librosa.stft(x, n_fft=N_FFT))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(DB, sr=fs, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pitch_contour(x, fs, title="Pitch Contour (f0)"):
    f0, voiced_flag, voiced_probs = librosa.pyin(x, fmin=70, fmax=400)
    plt.figure(figsize=(10, 2.5))
    plt.plot(f0, color='crimson', lw=1)
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Frame index")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------
# Main flow
# -----------------------
def main():
    print("=== Voice Morphing Lab (Enhanced) ===")
    print(f"Recording {DURATION} seconds at {FS} Hz...")
    time.sleep(0.8)

    # Record
    recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    x = recording[:, 0].astype(np.float32)

    # Show raw plots (blocking)
    print("Showing raw recorded waveform... (close the figure to continue)")
    plot_waveform(x, FS, title="Raw Recorded Waveform")
    print("Showing raw spectrogram... (close the figure to continue)")
    plot_spectrogram(x, FS, title="Raw Spectrogram")
    print("Showing raw spectrum... (close the figure to continue)")
    plot_spectrum(x, FS, title="Raw Magnitude Spectrum")
    print("Showing pitch contour (raw) ... (close figure to continue)")
    plot_pitch_contour(x, FS, title="Pitch Contour (Raw)")

    # Choose morph
    print("\nChoose morph style:")
    print(" 1) Female")
    print(" 2) Male")
    print(" 3) Robot")
    print(" 4) Deep/Alien")
    print(" 5) Chipmunk")
    choice = input("Enter (1-5): ").strip()

    if choice == "1":
        morph_fn = morph_female; label = "female"
    elif choice == "2":
        morph_fn = morph_male; label = "male"
    elif choice == "3":
        morph_fn = morph_robot; label = "robot"
    elif choice == "4":
        morph_fn = morph_deep; label = "deep"
    elif choice == "5":
        morph_fn = morph_chipmunk; label = "chipmunk"
    else:
        morph_fn = lambda a, b: a; label = "none"

    # Pre-filter
    x_filt = butter_bandpass(x, 80, 8000, FS, order=4)

    # Show filtered plots
    print("Showing filtered waveform... (close to continue)")
    plot_waveform(x_filt, FS, title="Filtered Waveform (80-8kHz)")
    print("Showing filtered spectrum... (close to continue)")
    plot_spectrum(x_filt, FS, title="Filtered Magnitude Spectrum")
    print("Showing pitch contour (filtered) ... (close figure to continue)")
    plot_pitch_contour(x_filt, FS, title="Pitch Contour (Filtered)")

    # Morph (progress)
    print(f"\nApplying morph: {label} — this may take a moment...")
    for _ in tqdm(range(60), desc="Processing", ncols=70):
        time.sleep(0.01)

    y = morph_fn(x_filt, FS)
    y = normalize(y)

    # Show after-morph visuals (blocking)
    print("Showing morphed waveform... (close to continue)")
    plot_waveform(y, FS, title=f"Morphed Waveform ({label})")

    print("Showing morphed spectrum... (close to continue)")
    plot_spectrum(y, FS, title=f"Morphed Magnitude Spectrum ({label})")

    print("Showing morphed spectrogram... (close to continue)")
    plot_spectrogram(y, FS, title=f"Morphed Spectrogram ({label})")

    print("Showing pitch contour comparison... (close to continue)")
    # pitch before and after
    f0_before, _, _ = librosa.pyin(x_filt, fmin=70, fmax=400)
    f0_after, _, _  = librosa.pyin(y, fmin=70, fmax=400)
    plt.figure(figsize=(10, 3))
    plt.plot(f0_before, label='before', alpha=0.8)
    plt.plot(f0_after, label='after', alpha=0.8)
    plt.title("Pitch Contour Comparison (before vs after)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save and play
    out_path = f"morphed_{label}_{int(time.time())}.wav"
    sf.write(out_path, y, FS)
    print(f"\nSaved morphed file: {os.path.abspath(out_path)}")

    print("Playing morphed audio...")
    sd.play(y, FS)
    sd.wait()
    print("Playback done.")

    print("\nALL DONE — check the saved file and the plots you closed.")

if __name__ == "__main__":
    main()