import os
import wave
import numpy as np
import matplotlib.pyplot as plt

def graph_spectrogram(wav_file, figure_name):
    # Check if the image already exists
    if os.path.exists(figure_name):
        print(f"Spectrogram already exists: {figure_name}")
        return

    # Get WAV file information
    sound_info, frame_rate = get_wav_info(wav_file)

    # Plot the spectrogram
    plt.figure(figsize=(0.84, 0.84))
    plt.axis('off')
    plt.specgram(sound_info, Fs=frame_rate)
    plt.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory

def get_wav_info(wav_file):
    with wave.open(wav_file, 'r') as wav:
        frames = wav.readframes(-1)
        sound_info = np.frombuffer(frames, dtype=np.int16)
        frame_rate = wav.getframerate()
    return sound_info, frame_rate
