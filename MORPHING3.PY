import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
from scipy.signal import butter, lfilter
from tqdm import tqdm
import time
import os

# -----------------------
# PARAMETERS
# -----------------------
fs = 44100               # Sampling rate
duration = 8             # seconds
lowcut = 80              # Bandpass low cutoff
highcut = 4000           # Bandpass high cutoff
pitch_shift_value = 4    # Semitones to shift (+ for high pitch, - for low)
output_filename = "morphed_voice.wav"

# -----------------------
# BANDPASS FILTER FUNCTION
# -----------------------
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    return lfilter(b, a, data)

# -----------------------
# STAGE 1: RECORDING
# -----------------------
print("\n🎤 Starting Voice Recording...")
print(f"Speak now for {duration} seconds!")

recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')

# Live waveform display
plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("🎧 Live Recording Waveform")
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Amplitude")
line, = ax.plot(np.zeros(1000))
plt.show()

for i in range(duration * 10):
    chunk = recording[:int((i+1)*fs/10)]
    if len(chunk) > 0:
        data = chunk.flatten()
        if len(data) < 1000:
            data = np.pad(data, (0, 1000-len(data)), 'constant')
        line.set_ydata(data)
        plt.pause(0.1)

sd.wait()
plt.close()
print("✅ Recording Complete!\n")

# -----------------------
# STAGE 2: FILTERING
# -----------------------
print("🎚️ Applying bandpass filter...")
audio_filtered = butter_bandpass_filter(recording[:, 0], lowcut, highcut, fs)

plt.figure(figsize=(8,4))
plt.title("Filtered Voice Signal")
plt.plot(audio_filtered, color='teal')
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show(block=False)
plt.pause(1.5)
plt.close()

# -----------------------
# STAGE 3: FEATURE ANALYSIS
# -----------------------
print("📈 Extracting pitch contour...")
f0, voiced_flag, voiced_probs = librosa.pyin(audio_filtered, fmin=80, fmax=400)
avg_pitch = np.nanmean(f0)
print(f"Average Pitch Before Morphing: {avg_pitch:.2f} Hz")

plt.figure(figsize=(8,4))
plt.title("Pitch Contour of Original Voice")
plt.plot(f0, color='darkred')
plt.xlabel("Frame index")
plt.ylabel("Pitch (Hz)")
plt.show(block=False)
plt.pause(1.5)
plt.close()

# -----------------------
# STAGE 4: VOICE MORPHING
# -----------------------
print("🎨 Morphing your voice...")

# Ensure data is in float32 format
audio_filtered = audio_filtered.astype(np.float32)

# Pitch shifting
shifted_audio = librosa.effects.pitch_shift(audio_filtered, sr=fs, n_steps=pitch_shift_value)

# Optional: time stretch for added effect
shifted_audio = librosa.effects.time_stretch(shifted_audio, rate=1.1)

# Normalize output to prevent clipping
shifted_audio = shifted_audio / np.max(np.abs(shifted_audio))

# Show spectrogram
plt.figure(figsize=(8,4))
D = librosa.amplitude_to_db(np.abs(librosa.stft(shifted_audio)), ref=np.max)
librosa.display.specshow(D, sr=fs, x_axis='time', y_axis='log', cmap='magma')
plt.title("Spectrogram of Morphed Voice")
plt.colorbar(format="%+2.0f dB")
plt.show(block=False)
plt.pause(1.5)
plt.close()

# -----------------------
# STAGE 5: SAVE OUTPUT
# -----------------------
sf.write(output_filename, shifted_audio, fs)
print(f"💾 Morphed voice saved as '{output_filename}' ✅")

# -----------------------
# STAGE 6: PLAYBACK
# -----------------------
print("\n🎧 Playing your morphed voice...")
sd.play(shifted_audio, fs)
for _ in tqdm(range(int(len(shifted_audio)/fs))):
    time.sleep(1)
sd.wait()
print("Playback finished ✅")

# -----------------------
# STAGE 7: SUMMARY
# -----------------------
print("\n🧾 Summary:")
print(f"🎙️ Recording Duration: {duration}s")
print(f"🎵 Pitch Shift Applied: {pitch_shift_value:+} semitones")
print(f"📊 Average Pitch Before: {avg_pitch:.2f} Hz")
print(f"💽 Output File: {os.path.abspath(output_filename)}")
print("\n✨ Voice Morphing Completed Successfully! ✨")