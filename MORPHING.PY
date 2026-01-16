import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
from scipy.signal import butter, lfilter
import time, os, sys
from tqdm import tqdm

# --------------------------
# PARAMETERS
# --------------------------
fs = 44100
duration = 8
output_filename = "morphed_voice.wav"

# --------------------------
# FILTER FUNCTION
# --------------------------
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return lfilter(b, a, data)

# --------------------------
# TERMINAL VISUAL EFFECT
# --------------------------
def hacking_effect(text, delay=0.02):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    print("")

# --------------------------
# RECORDING
# --------------------------
hacking_effect("\n🎤 Booting up Neural Voice Interface...")
time.sleep(0.8)
print(f"🎙️ Speak for {duration} seconds...")

recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("✅ Recording complete!\n")

# --------------------------
# VISUAL 1: Raw Signal
# --------------------------
plt.figure(figsize=(10, 4))
plt.plot(recording, color='violet')
plt.title("Raw Recorded Signal (Time Domain)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show(block=False)
plt.pause(1.2)
plt.close()

# --------------------------
# CHOICE OF MORPH
# --------------------------
print("Select your morphing style:")
print("1️⃣  Female Voice")
print("2️⃣  Male Voice")
print("3️⃣  Robot Voice")
print("4️⃣  Alien / Deep Resonance")
choice = input("Enter choice (1-4): ")

if choice == "1":
    pitch_shift = +6; rate = 1.2; desc = "👩 Female Voice"
elif choice == "2":
    pitch_shift = -4; rate = 0.9; desc = "👨 Male Voice"
elif choice == "3":
    pitch_shift = 0; rate = 1.0; desc = "🤖 Robotic Voice"
elif choice == "4":
    pitch_shift = -7; rate = 0.8; desc = "👽 Alien Resonance"
else:
    pitch_shift = 0; rate = 1.0; desc = "Default Morph"

# --------------------------
# FILTERING
# --------------------------
hacking_effect("\n🎚️ Filtering low & high noise frequencies...")
audio_filtered = butter_bandpass_filter(recording[:, 0], 80, 4000, fs)

# Plot Filtered Signal
plt.figure(figsize=(10, 4))
plt.plot(audio_filtered, color='lime')
plt.title("Filtered Signal (After Bandpass Filter)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show(block=False)
plt.pause(1.2)
plt.close()

# Plot FFT of Filtered Signal
plt.figure(figsize=(10, 4))
fft_vals = np.abs(np.fft.rfft(audio_filtered))
fft_freqs = np.fft.rfftfreq(len(audio_filtered), 1/fs)
plt.plot(fft_freqs, fft_vals, color='cyan')
plt.title("Frequency Spectrum (Filtered Signal)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show(block=False)
plt.pause(1.2)
plt.close()

# --------------------------
# PITCH CONTOUR BEFORE MORPHING
# --------------------------
f0_before, _, _ = librosa.pyin(audio_filtered, fmin=80, fmax=400)
plt.figure(figsize=(10, 4))
plt.plot(f0_before, color='red')
plt.title("Pitch Contour (Before Morphing)")
plt.xlabel("Frames")
plt.ylabel("Pitch (Hz)")
plt.grid(True)
plt.show(block=False)
plt.pause(1.2)
plt.close()

# --------------------------
# MORPHING
# --------------------------
hacking_effect(f"\n{desc} transformation in progress...")
for _ in tqdm(range(100), desc="⚙️ Processing Neural Layers", ncols=80):
    time.sleep(0.015)

shifted = librosa.effects.pitch_shift(audio_filtered, sr=fs, n_steps=pitch_shift)
shifted = librosa.effects.time_stretch(shifted, rate=rate)
shifted = shifted / np.max(np.abs(shifted))

# --------------------------
# VISUALS: AFTER MORPHING
# --------------------------
plt.figure(figsize=(10, 4))
plt.plot(shifted, color='orange')
plt.title("Morphed Signal (Time Domain)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show(block=False)
plt.pause(1.2)
plt.close()

# FFT after morphing
plt.figure(figsize=(10, 4))
fft_vals2 = np.abs(np.fft.rfft(shifted))
fft_freqs2 = np.fft.rfftfreq(len(shifted), 1/fs)
plt.plot(fft_freqs2, fft_vals2, color='gold')
plt.title("Frequency Spectrum (After Morphing)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show(block=False)
plt.pause(1.2)
plt.close()

# Spectrogram Before & After
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
S1 = librosa.amplitude_to_db(np.abs(librosa.stft(audio_filtered)), ref=np.max)
S2 = librosa.amplitude_to_db(np.abs(librosa.stft(shifted)), ref=np.max)
librosa.display.specshow(S1, sr=fs, x_axis='time', y_axis='log', cmap='magma', ax=axs[0])
axs[0].set_title("Spectrogram (Before Morphing)")
librosa.display.specshow(S2, sr=fs, x_axis='time', y_axis='log', cmap='viridis', ax=axs[1])
axs[1].set_title("Spectrogram (After Morphing)")
plt.tight_layout()
plt.show(block=False)
plt.pause(2)
plt.close()

# Pitch comparison
f0_after, _, _ = librosa.pyin(shifted, fmin=80, fmax=400)
plt.figure(figsize=(10, 4))
plt.plot(f0_before, label="Before", color='crimson')
plt.plot(f0_after, label="After", color='springgreen')
plt.title("Pitch Contour Comparison")
plt.xlabel("Frames")
plt.ylabel("Pitch (Hz)")
plt.legend()
plt.grid(True)
plt.show(block=False)
plt.pause(2)
plt.close()

# --------------------------
# SAVE & PLAYBACK
# --------------------------
sf.write(output_filename, shifted, fs)
print(f"💾 Saved: {os.path.abspath(output_filename)}")

print("\n🎧 Playing morphed audio...")
sd.play(shifted, fs)
for _ in tqdm(range(int(len(shifted)/fs))):
    time.sleep(1)
sd.wait()
print("✅ Playback complete!\n")

# --------------------------
# SUMMARY
# --------------------------
print("🧾 SUMMARY")
print(f"🎵 Mode: {desc}")
print(f"🔺 Pitch Shift: {pitch_shift:+} semitones")
print(f"⏱️ Time Stretch: {rate}")
print(f"💽 File: {os.path.abspath(output_filename)}")
print("\n✨ All DSP visualizations generated successfully! ✨")