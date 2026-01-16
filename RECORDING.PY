import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
from scipy.signal import butter, lfilter

# ------------------------
# PARAMETERS
# ------------------------
fs = 44100        # Sampling rate
duration = 15     # seconds
filename = "original_voice.wav"

# ------------------------
# STEP 1: RECORDING
# ------------------------
print("🎙️ Get ready! Recording starts in 2 seconds...")
sd.sleep(2000)
print("🎤 Recording started for 15 seconds... Speak now!")

audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("✅ Recording complete! Saved as:", filename)
sf.write(filename, audio, fs)

# ------------------------
# STEP 2: VISUALIZE RECORDED AUDIO
# ------------------------
plt.figure(figsize=(10, 4))
plt.plot(audio, color='cyan')
plt.title("Original Recorded Voice Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Spectrogram
plt.figure(figsize=(10, 4))
D = librosa.amplitude_to_db(np.abs(librosa.stft(audio[:, 0])), ref=np.max)
librosa.display.specshow(D, sr=fs, x_axis='time', y_axis='log', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram of Original Voice")
plt.show()

# ------------------------
# STEP 3: MORPHING OPTIONS
# ------------------------
print("\n🎨 Choose how you want to morph your voice:")
print("1️⃣ Male voice (deeper, slower)")
print("2️⃣ Female voice (higher pitch)")
print("3️⃣ Chipmunk (very high pitch, fast)")
print("4️⃣ Robot (metallic effect)")
print("5️⃣ Echo effect")
choice = input("Enter your choice (1-5): ")

# ------------------------
# STEP 4: PROCESSING (MORPHING)
# ------------------------

# Helper: Bandpass filter
def butter_bandpass_filter(data, lowcut=80, highcut=4000, fs=44100, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    return lfilter(b, a, data)

# Apply a basic bandpass first
audio_filtered = butter_bandpass_filter(audio[:, 0])

# Voice morphing based on user choice
if choice == "1":
    # Male
    morphed = librosa.effects.pitch_shift(audio_filtered, sr=fs, n_steps=-4)
    morphed = librosa.effects.time_stretch(morphed, rate=0.9)
    effect = "Male Voice"
elif choice == "2":
    # Female
    morphed = librosa.effects.pitch_shift(audio_filtered, sr=fs, n_steps=4)
    morphed = librosa.effects.time_stretch(morphed, rate=1.1)
    effect = "Female Voice"
elif choice == "3":
    # Chipmunk
    morphed = librosa.effects.pitch_shift(audio_filtered, sr=fs, n_steps=8)
    morphed = librosa.effects.time_stretch(morphed, rate=1.3)
    effect = "Chipmunk Voice"
elif choice == "4":
    # Robot-like metallic tone
    echo = np.roll(audio_filtered, 1000)
    morphed = (audio_filtered + 0.5 * echo)
    effect = "Robot Voice"
elif choice == "5":
    # Echo effect
    delay = 12000
    echo = np.zeros_like(audio_filtered)
    echo[delay:] = audio_filtered[:-delay]
    morphed = audio_filtered + 0.6 * echo
    effect = "Echo Voice"
else:
    morphed = audio_filtered
    effect = "Original Voice (no change)"

# Normalize to prevent clipping
morphed = morphed / np.max(np.abs(morphed))

# ------------------------
# STEP 5: VISUALIZE MORPHED VOICE
# ------------------------
plt.figure(figsize=(10, 4))
plt.plot(morphed, color='orange')
plt.title(f"Morphed Voice Waveform ({effect})")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

D2 = librosa.amplitude_to_db(np.abs(librosa.stft(morphed)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D2, sr=fs, x_axis='time', y_axis='log', cmap='plasma')
plt.colorbar(format='%+2.0f dB')
plt.title(f"Spectrogram of {effect}")
plt.show()

# ------------------------
# STEP 6: SAVE + PLAYBACK
# ------------------------
output_filename = f"morphed_{effect.replace(' ', '_').lower()}.wav"
sf.write(output_filename, morphed, fs)
print(f"✅ Morphed audio saved as: {output_filename}")

print("\n🎧 Playing the morphed voice...")
sd.play(morphed, fs)
sd.wait()
print("Playback finished ✅")

print("\n✨ Voice morphing completed successfully!")