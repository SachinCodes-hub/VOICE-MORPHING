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
duration = 20
output_filename = "morphed_voice.wav"

# --------------------------
# FILTER FUNCTION
# --------------------------
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import butter, lfilter
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return lfilter(b, a, data)

# --------------------------
# REALTIME CONSOLE EFFECT
# --------------------------
def hacking_effect(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print("")

# --------------------------
# RECORDING
# --------------------------
hacking_effect("\n🎤 Initializing Quantum Voice Capture System...")
time.sleep(1)
print(f"🎙️ Speak for {duration} seconds...")

recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("✅ Recording complete!\n")

# --------------------------
# MORPH CHOICE
# --------------------------
print("Select your morphing style:")
print("1️⃣  Female Voice")
print("2️⃣  Male Voice")
print("3️⃣  Robot Voice")
print("4️⃣  Alien / Deep Resonance")
choice = input("Enter choice (1-4): ")

# Set morphing parameters based on choice
if choice == "1":
    pitch_shift = +6
    rate = 1.2
    desc = "👩 Transforming into Female Voice"
elif choice == "2":
    pitch_shift = -4
    rate = 0.9
    desc = "👨 Transforming into Male Voice"
elif choice == "3":
    pitch_shift = 0
    rate = 1.0
    desc = "🤖 Synthesizing Robotic Tone"
elif choice == "4":
    pitch_shift = -7
    rate = 0.8
    desc = "👽 Generating Alien Resonance"
else:
    pitch_shift = 0
    rate = 1.0
    desc = "Default Morph"

# --------------------------
# FILTERING
# --------------------------
hacking_effect("\n🎚️ Filtering out noise frequencies...")
audio_filtered = butter_bandpass_filter(recording[:, 0], 80, 4000, fs)

plt.figure(figsize=(10, 4))
plt.plot(audio_filtered, color='lime')
plt.title("Filtered Voice Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show(block=False)
plt.pause(1.2)
plt.close()

# --------------------------
# VISUALIZE FFT BEFORE MORPHING
# --------------------------
plt.figure(figsize=(10, 4))
fft_vals = np.abs(np.fft.rfft(audio_filtered))
fft_freqs = np.fft.rfftfreq(len(audio_filtered), 1/fs)
plt.plot(fft_freqs, fft_vals, color='cyan')
plt.title("Frequency Spectrum Before Morphing")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show(block=False)
plt.pause(1.2)
plt.close()

# --------------------------
# MORPHING STAGE
# --------------------------
hacking_effect(f"\n{desc}...")
time.sleep(0.5)
for _ in tqdm(range(100), desc="⚙️ Morphing Signal Matrix", ncols=80):
    time.sleep(0.02)

# Pitch and time transform
shifted = librosa.effects.pitch_shift(audio_filtered, sr=fs, n_steps=pitch_shift)
shifted = librosa.effects.time_stretch(shifted, rate=rate)
shifted = shifted / np.max(np.abs(shifted))

# --------------------------
# DSP VISUALS DURING MORPHING
# --------------------------
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
plt.suptitle("🔊 DSP Transformation Visuals", fontsize=14, color='gold')

# Waveform
axs[0].plot(shifted[:5000], color='orange')
axs[0].set_title("Waveform After Morphing")
axs[0].set_ylabel("Amplitude")

# Spectrogram
S = np.abs(librosa.stft(shifted))
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=fs, x_axis='time', y_axis='log', ax=axs[1], cmap='magma')
axs[1].set_title("Spectrogram (log scale)")

# Pitch contour
f0, voiced_flag, voiced_probs = librosa.pyin(shifted, fmin=80, fmax=400)
axs[2].plot(f0, color='cyan')
axs[2].set_title("Pitch Contour (Morphed Voice)")
axs[2].set_xlabel("Frames")
axs[2].set_ylabel("Pitch (Hz)")

plt.tight_layout()
plt.show(block=False)
plt.pause(2.5)
plt.close()

# --------------------------
# SAVE & PLAYBACK
# --------------------------
sf.write(output_filename, shifted, fs)
print(f"💾 Saved morphed voice: {os.path.abspath(output_filename)}")

print("\n🎧 Playing morphed voice...")
sd.play(shifted, fs)
for _ in tqdm(range(int(len(shifted)/fs))):
    time.sleep(1)
sd.wait()
print("✅ Playback finished!")

# --------------------------
# SUMMARY
# --------------------------
print("\n🧾 Summary:")
print(f"🎵 Mode: {desc}")
print(f"🔺 Pitch Shift: {pitch_shift:+} semitones")
print(f"⏱️ Time Stretch: {rate}")
print(f"💽 Output: {os.path.abspath(output_filename)}")
print("\n✨ Voice Morphing Completed with Visual DSP Magic! ✨")