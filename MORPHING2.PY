import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import queue, sys, time

# ---- SETTINGS ----
duration = 5  # seconds
fs = 44100  # Hz
blocksize = 1024

q = queue.Queue()
print("🎤 Preparing for recording...")

# ---- AUDIO CALLBACK ----
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

# ---- VISUAL SETUP ----
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
fig.patch.set_facecolor('#0a0a0a')

# Waveform
x = np.arange(0, blocksize)
y = np.zeros(blocksize)
waveform_line, = axs[0].plot(x, y, color='cyan', lw=2)
axs[0].set_xlim(0, blocksize)
axs[0].set_ylim(-1, 1)
axs[0].set_title("🎶 Live Waveform", color='white')
axs[0].set_facecolor('#121212')
axs[0].tick_params(colors='white')

# Spectrum Analyzer
fft_x = np.fft.rfftfreq(blocksize, 1/fs)
fft_y = np.zeros(len(fft_x))
bars = axs[1].bar(fft_x, fft_y, width=20, color='magenta')
axs[1].set_xlim(0, 8000)
axs[1].set_ylim(0, 50)
axs[1].set_title("⚡ Real-Time Spectrum", color='white')
axs[1].set_facecolor('#121212')
axs[1].tick_params(colors='white')

# Progress / VU Meter
axs[2].set_xlim(0, duration)
axs[2].set_ylim(0, 1)
progress_bar, = axs[2].plot([], [], color='lime', lw=4)
axs[2].set_title("🎧 Recording Progress + Volume Level", color='white')
axs[2].set_facecolor('#121212')
axs[2].tick_params(colors='white')

start_time = time.time()
progress_times, vu_levels = [], []

# ---- UPDATE ANIMATION ----
def update_plot(frame):
    global progress_times, vu_levels

    while not q.empty():
        data = q.get_nowait().flatten()

        # Fix shape mismatch: always resize data to match the plot length
        if len(data) != len(y):
            data = np.interp(np.linspace(0, len(data), len(y)), np.arange(len(data)), data)

        # Update waveform
        waveform_line.set_ydata(data)

        # Update FFT bars
        fft_vals = np.abs(np.fft.rfft(data))
        fft_vals = np.log10(fft_vals + 1)
        for rect, h in zip(bars, fft_vals[:len(bars)]):
            rect.set_height(h)

        # Update VU level + progress
        current_time = time.time() - start_time
        rms = np.sqrt(np.mean(data**2))
        progress_times.append(current_time)
        vu_levels.append(rms)
        progress_bar.set_data(progress_times, vu_levels)

    return waveform_line, bars, progress_bar

# ---- RECORDING + VISUALS ----
print("🎙️ Recording started — enjoy the visuals!")
with sd.InputStream(callback=audio_callback, channels=1, samplerate=fs, blocksize=blocksize):
    ani = animation.FuncAnimation(fig, update_plot, interval=30, blit=False)
    plt.tight_layout()
    plt.show(block=False)
    time.sleep(duration)
print("✅ Recording finished!")

# ---- PLAYBACK ----
print("\n🔊 Playing back your recording...")
recorded_audio = []
while not q.empty():
    recorded_audio.append(q.get())
if recorded_audio:
    recorded_audio = np.concatenate(recorded_audio, axis=0)
    sd.play(recorded_audio, fs)
    sd.wait()

# ---- FINAL VISUAL SUMMARY ----
plt.figure(figsize=(10, 5))
plt.plot(recorded_audio, color='cyan', lw=1)
plt.title("✨ Final Recorded Waveform ✨", color='white', fontsize=14)
plt.gca().set_facecolor("#121212")
plt.gca().tick_params(colors='white')
plt.grid(alpha=0.3)
plt.show()

print("🎉 Done! Your audio has been visualized and played successfully.")