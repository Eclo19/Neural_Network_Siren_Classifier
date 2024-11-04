import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import sounddevice as sd
import scipy.signal as signal
import librosa

"""
This file contains implementations of simple mono plugins. These are meant to be used in the training
data (with light to medium settings, when applicable) to artificially expand the dataset. For plotting and playing, 
all functions have two types: 'data' and 'path'. As the names suggest, 'data' type interprets the input as raw audio data and 
'path' interprets the input as a string with a path to the data in the disk. Tests calls the plotting and playing functions to 
look at the plugins' effects on data. 'stereo_to_mono' takes in a stereo file (most training datafiles are stereo) and 
generates two mono_files (L and R) or sums both channels into one mono file If the file is mono, it leaves it unchanged with a 
printed warning. Furthermore, for compatibility with some librosa processing, some plugins need to work with float32 instead of 
the int16 that spawns from WAVE files with a bit depth of 16. 
"""


#Global variables
BIT_DEPTH = 16
SAMPLE_RATE = 44100

#Plugins:

def compressor(audio, threshold, ratio, format="int16", make_up_gain=True):
    """
    Apply a basic naive compressor to the audio signal and return the processed signal
    based on the specified format ('float32' or 'int16').
    
    Parameters:
    - audio: NumPy array containing the audio signal (either float32 or int16).
    - threshold: Threshold in dB above which the signal will be compressed.
    - ratio: Compression ratio applied to samples larger than the threshold.
    - format: The desired format for input/output ('float32' or 'int16').
    - make_up_gain: If True, apply make-up gain after compression to restore average loudness.
    
    Returns:
    - NumPy array containing the compressed audio signal in the same format as input.

    Modifies:
    - Nothing 
    """
    # Convert threshold from dB to linear scale
    threshold_linear = 10 ** (threshold / 20)

    # Handle different formats
    if format == "int16":

        # Convert threshold to match int16 range
        threshold_linear *= 32767
    
    elif format == "float32":
        # Ensure audio is in the range [-1, 1]
        threshold_linear = np.clip(threshold_linear, -1, 1)

    # Initialize compressed audio as a copy of the original audio
    comp_audio = np.copy(audio)

    # Apply compression
    comp_audio = np.where(np.abs(audio) > threshold_linear, 
                          np.sign(audio) * (threshold_linear + (np.abs(audio) - threshold_linear) / ratio), 
                          audio)

    # Apply make-up gain if requested
    if make_up_gain:
        avg_level_before = np.mean(np.abs(audio))
        avg_level_after = np.mean(np.abs(comp_audio))
        if avg_level_after > 0:  # Prevent division by zero
            gain_factor = avg_level_before / avg_level_after
            comp_audio *= gain_factor

    # Handle clipping and conversion for int16 format
    if format == "int16":
        comp_audio = np.clip(comp_audio, -32768, 32767).astype(np.int16)
    
    # For float32, ensure the output remains within the range [-1, 1] without clipping excessively
    elif format == "float32":
        comp_audio = np.clip(comp_audio, -1, 1).astype(np.float32)

    return comp_audio


def shelving(x, G, fc, fs, Q, filter_type='low', plot=False):
    """
    Derive coefficients for a shelving filter with a given amplitude and cutoff frequency.

    Parameters:
    G           : Logarithmic gain (in dB)
    fc          : Cutoff frequency (Hz)
    fs          : Sampling rate (Hz)
    Q           : Adjusts the slope (Quality factor)
    filter_type : 'low' or 'high'

    Returns:
    b, a : Numerator (b) and denominator (a) filter coefficients

    Modifies: 
    - Nothing
    """

    if filter_type not in ['low', 'high']:
        raise ValueError(f"Unsupported filter type: {filter_type}. Only 'high' and 'low' are possible.")

    K = np.tan(np.pi * fc / fs)
    V0 = 10 ** (G / 20)
    root2 = 1 / Q  # Equivalent to sqrt(2) in shelving filter formulas

    # Invert gain if it's a cut (V0 < 1)
    if V0 < 1:
        V0 = 1 / V0

    if filter_type == 'low':
        if G > 0:  # Bass boost
            b0 = (1 + np.sqrt(V0) * root2 * K + V0 * K ** 2) / (1 + root2 * K + K ** 2)
            b1 = (2 * (V0 * K ** 2 - 1)) / (1 + root2 * K + K ** 2)
            b2 = (1 - np.sqrt(V0) * root2 * K + V0 * K ** 2) / (1 + root2 * K + K ** 2)
            a1 = (2 * (K ** 2 - 1)) / (1 + root2 * K + K ** 2)
            a2 = (1 - root2 * K + K ** 2) / (1 + root2 * K + K ** 2)
        else:  # Bass cut
            b0 = (1 + root2 * K + K ** 2) / (1 + root2 * np.sqrt(V0) * K + V0 * K ** 2)
            b1 = (2 * (K ** 2 - 1)) / (1 + root2 * np.sqrt(V0) * K + V0 * K ** 2)
            b2 = (1 - root2 * K + K ** 2) / (1 + root2 * np.sqrt(V0) * K + V0 * K ** 2)
            a1 = (2 * (V0 * K ** 2 - 1)) / (1 + root2 * np.sqrt(V0) * K + V0 * K ** 2)
            a2 = (1 - root2 * np.sqrt(V0) * K + V0 * K ** 2) / (1 + root2 * np.sqrt(V0) * K + V0 * K ** 2)

    elif filter_type == 'high':
        if G > 0:  # Treble boost
            b0 = (V0 + root2 * np.sqrt(V0) * K + K ** 2) / (1 + root2 * K + K ** 2)
            b1 = (2 * (K ** 2 - V0)) / (1 + root2 * K + K ** 2)
            b2 = (V0 - root2 * np.sqrt(V0) * K + K ** 2) / (1 + root2 * K + K ** 2)
            a1 = (2 * (K ** 2 - 1)) / (1 + root2 * K + K ** 2)
            a2 = (1 - root2 * K + K ** 2) / (1 + root2 * K + K ** 2)
        else:  # Treble cut
            b0 = (1 + root2 * K + K ** 2) / (V0 + root2 * np.sqrt(V0) * K + K ** 2)
            b1 = (2 * (K ** 2 - 1)) / (V0 + root2 * np.sqrt(V0) * K + K ** 2)
            b2 = (1 - root2 * K + K ** 2) / (V0 + root2 * np.sqrt(V0) * K + K ** 2)
            a1 = (2 * (K ** 2 / V0 - 1)) / (1 + root2 / np.sqrt(V0) * K + (K ** 2) / V0)
            a2 = (1 - root2 / np.sqrt(V0) * K + (K ** 2) / V0) / (1 + root2 / np.sqrt(V0) * K + (K ** 2) / V0)

    # Return the filter coefficients
    a = [1, a1, a2]
    b = [b0, b1, b2]

    y = signal.lfilter(b, a, x)

    if plot:
        # Plot the Bode plot (magnitude and phase response)
        w, h = signal.freqz(b, a, worN=2000, fs=fs)
        fig, ax1 = plt.subplots()
        
        # Magnitude plot
        ax1.set_title("Frequency Response (Magnitude and Phase)")
        ax1.plot(w, 20 * np.log10(np.abs(h)), 'b')
        ax1.set_xscale('log')
        ax1.set_ylabel('Gain [dB]', color='b')
        ax1.set_xlabel('Frequency [Hz]')
        ax1.grid(True)
        ax1.set_xlim([20, fs / 2])

        # Phase plot
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w, np.degrees(angles), 'g')
        ax2.set_ylabel('Phase [degrees]', color='g')
        ax2.grid(True)

        plt.show()

    return y


def pitch_shift_fx(audio, sample_rate, pitch_shift_steps):
    """
    Apply pitch shifting to an audio signal.

    Parameters:
    - audio: NumPy array containing the audio signal. It must be of type float32.
    - sample_rate: Sampling rate of the audio signal (e.g., 44100).
    - pitch_shift_steps: Number of steps to shift the pitch (positive for up, negative for down).

    Returns:
    - NumPy array containing the pitch-shifted audio signal, converted to int16 with the same sample rate.

    Modifies:
    - Nothing 
    
    Raises:
    - ValueError: If the input audio is not of type float32.
    """
    # Ensure input audio is in float32 format
    if audio.dtype != np.float32:
        raise ValueError("Input audio must be of type float32.")

    try:
        # Apply pitch shift using librosa's pitch_shift function
        pitch_shifted_audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift_steps)

        # Clip the values to the valid range for int16 (-32768 to 32767)
        pitch_shifted_audio = np.clip(pitch_shifted_audio, -1.0, 1.0)

        # Convert the float32 array to int16
        pitch_shifted_audio_int16 = (pitch_shifted_audio * 32767).astype(np.int16)

        return pitch_shifted_audio_int16

    except Exception as e:
        print(f"An error occurred while pitch shifting: {e}")
        return None


def mask_audio(input_data, sr, mask_type='frequency', max_mask_pct=0.1, plot=False):
    """
    Apply a frequency or time mask to the audio, handling both file paths and audio data.

    Parameters:
    - input_data  : Either a NumPy array of floating point audio data or a string path to the file.
    - sr          : Sampling rate.
    - mask_type   : 'frequency' or 'time' mask type.
    - max_mask_pct: Maximum percentage of the spectrogram to mask.
    - plot        : Whether to plot the original and masked spectrograms.
    
    Returns:
    - masked_audio: The masked floating point audio data.

    Returns:
    - Nothing
    """

    def apply_mask(audio, sr, mask_type, max_mask_pct=0.1, plot=False):
        """
        Apply a mask (frequency or time) to the spectrogram of the given audio.
        
        Parameters:
        - audio       : NumPy array containing the audio data or a file path to the audio file.
        - sr          : Sampling rate of the audio.
        - mask_type   : Type of mask to apply ('frequency' or 'time').
        - max_mask_pct: Maximum percentage of the spectrogram to mask (default is 10%).
        - plot        : Whether to plot the original and masked spectrograms.
        
        Returns:
        - masked_audio: The masked audio data as a floating point NumPy array.

        Modifies: Nothing 
        """

        # Convert audio to a spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        # Get dimensions of the spectrogram
        num_freqs, num_frames = spectrogram_db.shape

        if mask_type == "frequency":
            mask_size = np.random.randint(0, int(max_mask_pct * num_freqs))
            mask_start = np.random.randint(0, num_freqs - mask_size)
            spectrogram_db[mask_start:mask_start + mask_size, :] = 0

        elif mask_type == "time":
            mask_size = np.random.randint(0, int(max_mask_pct * num_frames))
            mask_start = np.random.randint(0, num_frames - mask_size)
            spectrogram_db[:, mask_start:mask_start + mask_size] = 0

        else:
            raise ValueError(f"Unsupported mask type: {mask_type}. Use 'frequency' or 'time'.")

        # Optionally plot the spectrogram before and after masking
        if plot:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
            plt.title('Masked Spectrogram')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.show()

        # Convert back to the audio signal from the masked spectrogram
        masked_audio = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sr)

        return masked_audio
    
    # If a file path is given, load the audio file
    if isinstance(input_data, str):
        audio, sr = librosa.load(input_data, sr=sr)
    else:
        audio = input_data

    # Apply the desired mask type to the audio data
    masked_audio = apply_mask(audio, sr, mask_type=mask_type, max_mask_pct=max_mask_pct, plot=plot)

    return masked_audio

#Stereo To Mono
def stereo_to_mono(filename, folder, mode="split"):
    """ 
    Takes in a stereo file and either splits it into two mono WAV files 
    (left and right channels) or sums it to create one mono WAV file. If the
    given file is not stereo, save it in the given directory as is if it is not there already.
    
    Parameters:
    - filename: Path to the stereo file.
    - folder  : Path to the output folder.
    - mode    : Either 'split' (for left and right) or 'sum' (to combine channels).

    Returns:
    - Nothing

    Modifies:
    - Adds folders to supplied directory
    """

    # Ensure the output folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Read the stereo wav file
    sr, stereo_data = wavfile.read(filename)

    # Normalize both channels of stereo_data
    stereo_data = stereo_data.astype(np.float32)  # Convert to float for normalization
    max_abs_value = np.max(np.abs(stereo_data))  # Get the maximum absolute value
    if max_abs_value > 0:  # Avoid division by zero
        stereo_data /= max_abs_value  # Normalize to [-1, 1]
        stereo_data *= 32767
        stereo_data = stereo_data.astype(np.int16)
    else:
        print("Warning: Maximum absolute value is zero, skipping normalization.")

    # Check if it's actually stereo (i.e., two channels)
    if len(stereo_data.shape) != 2 or stereo_data.shape[1] != 2:
        print(f"The input file '{filename}' is not a stereo file. Keeping it as is.")
        
        # Extract the original filename without the folder and extension
        base_filename = os.path.basename(filename)
        output_file_path = os.path.join(folder, base_filename)

        # Check if the file already exists in the output folder
        if not os.path.exists(output_file_path):
            # Save the original file to the specified folder
            wavfile.write(output_file_path, sr, stereo_data)
        else:
            print(f"The file '{output_file_path}' already exists. Not overwriting.")
        
        return
    
    # In 'split' mode, generate two files: left and right
    if mode == "split":
        # Split the stereo data into left and right channels
        left_channel = stereo_data[:, 0]
        right_channel = stereo_data[:, 1]

        # Extract the original filename without the folder and extension
        base_filename = os.path.basename(filename)
        name, ext = os.path.splitext(base_filename)

        # Construct filenames for left and right channels
        left_filename = os.path.join(folder, f"L_{name}{ext}")
        right_filename = os.path.join(folder, f"R_{name}{ext}")

        # Save the left and right channel as new mono files
        wavfile.write(left_filename, sr, left_channel)
        wavfile.write(right_filename, sr, right_channel)

    # In 'sum' mode, combine the left and right channels to mono
    elif mode == "sum":
        # Sum the left and right channels
        mono_data = np.mean(stereo_data, axis=1)

        # Normalize if necessary, making sure we stay within int16 range
        if stereo_data.dtype == np.int16:
            mono_data = np.clip(mono_data, -32768, 32767)
            mono_data = mono_data.astype(np.int16)
        elif stereo_data.dtype == np.float32 or stereo_data.dtype == np.float64:
            mono_data = np.clip(mono_data, -1.0, 1.0)
            mono_data = (mono_data * 32767).astype(np.int16)

        # Extract the original filename without the folder and extension
        base_filename = os.path.basename(filename)
        name, ext = os.path.splitext(base_filename)

        # Construct the filename for the mono file
        mono_filename = os.path.join(folder, f"Mono_{name}{ext}")

        # Save the mono file
        wavfile.write(mono_filename, sr, mono_data)

        print(f"Saved summed mono file to: {mono_filename}")
   
#Plotting and Playing

def plot_wav(input, type="path"):
    """
    Plots the waveform of a single WAV filepath or raw data. 
    If 'type' is 'path', it reads the file; if 'type' is 'data', it assumes the input is a tuple of (sr, NumPy array).
    """
    if type == "path":
        # Get data and sample rate from the file
        sr, data = wavfile.read(input)
    elif type == "data":
        # Assuming the input is a tuple (sample_rate, data)
        sr, data = input
    else:
        raise ValueError("Invalid type. Choose 'path' or 'data'.")

    # Convert to mono if stereo by averaging channels
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    # Generate time array
    time = np.linspace(0, len(data) / sr, num=len(data))

    plt.figure(figsize=(12, 8))
    plt.plot(time, data)
    plt.title(f"{input} Time-Domain Plot" if type == "path" else "Data Time-Domain Plot")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()



def plot_AB(data_A, data_B, mode="overlap", type="data"):
    """
    Plot two datasets (A and B) with a legend.
    Parameters:
    - data_A: Tuple (sample_rate_A, audio_data_A)
    - data_B: Tuple (sample_rate_B, audio_data_B)
    - mode: Plot mode, either "overlap" or "side_by_side"
    - type: Type of plot, either "data" (for waveform) or "spectrum" (for frequency domain)
    """
    sr_A, audio_A = data_A
    sr_B, audio_B = data_B

    plt.figure(figsize=(10, 6))

    if mode == "overlap":
        if type == "data":
            # Plot both waveforms on the same plot
            time_A = np.linspace(0, len(audio_A) / sr_A, num=len(audio_A))
            time_B = np.linspace(0, len(audio_B) / sr_B, num=len(audio_B))

            plt.plot(time_A, audio_A, label='A')  # Custom label A
            plt.plot(time_B, audio_B, label='B')  # Custom label B

            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title("Overlap of A and B")
            plt.legend(loc="upper right")  # Place the legend in a suitable position

    # Show the plot
    plt.show()


def play(input, type="path", volume=0.5):
    """
    Plays an audio file or raw data. Volume can be adjusted with the volume parameter (default is 1.0 for no change).
    If 'type' is 'path', it reads the file; if 'type' is 'data', it assumes the input is a tuple of (sr, NumPy array).
    Volume is a scalar multiplier to adjust loudness.
    """

    # Read the data based on the type
    if type == "path":
        sr, data = wavfile.read(input)
    elif type == "data":
        sr, data = input
    else:
        raise ValueError("Invalid type. Choose 'path' or 'data'.")

    # Normalize stereo to mono, if applicable
    if len(data.shape) == 2:
        data = data[:, 1]

    # Adjust volume
    data = data * volume

    # Normalize the data if necessary (to avoid clipping)
    max_val = np.max(np.abs(data))
    if max_val > 1.0:
        data = data / max_val

    # Play the audio
    sd.play(data, samplerate=sr, )
    sd.wait()  # Wait until the audio is done playing


def tests():

    #Load a test file from the vanilla dataset
    current_file_path = os.path.abspath(__file__)
    src_dir_path = os.path.dirname(current_file_path)
    project_root = os.path.dirname(src_dir_path)
    data_path = os.path.join(project_root, "data")
    filepath = os.path.join(data_path, "traffic")
    filename = "sound_403.wav"
    full_path = os.path.join(filepath, filename)

    #Get raw data
    sr, test_audio = wavfile.read(full_path)
    #Get right channel only, as the plugins are all mono
    test_audio_int16 = test_audio[:, 1]
    #Audio is in 16-bit PCM (WAV), equivalent to np.int16. We want it in float-point for librosa processing
    test_audio = test_audio_int16.astype(np.float32) / np.max(np.abs(test_audio))  # Normalize to [-1.0, 1.0]

    plot_wav((sr, test_audio), type="data")
    play(full_path, volume=0.3)

                                                        #COMPRESSOR: 

    #Try limiter
    comp_test_audio = compressor(test_audio_int16, -10, 2, format="int16", make_up_gain=True) 
    plot_AB((SAMPLE_RATE, test_audio), (SAMPLE_RATE, comp_test_audio), mode="overlap", type="data")

    #Play normal and limited
    play(full_path, volume=0.3)
    play((SAMPLE_RATE, comp_test_audio), volume=0.3, type="data")

                                                        #Shelving EQ: 


    #Apply shelfs and plot both audio files
    G = 4
    fc = 500
    Q = 1/np.sqrt(2)

    #High Shelf
    high_shelved_test_audio = shelving(test_audio, G, fc, SAMPLE_RATE, Q, filter_type='high')
    plot_AB((SAMPLE_RATE, test_audio), (SAMPLE_RATE, high_shelved_test_audio), mode="overlap", type="data")

    #Play normal and high shelf
    play((sr, test_audio), volume=0.3, type="data")
    play((SAMPLE_RATE, high_shelved_test_audio), volume=0.3, type="data")

    #Low Shelf
    low_shelved_test_audio = shelving(test_audio, G, fc, SAMPLE_RATE, Q, filter_type='low')
    plot_AB((SAMPLE_RATE, test_audio), (SAMPLE_RATE, low_shelved_test_audio), mode="overlap", type="data")

    #Play normal and high shelf
    play((sr, test_audio), volume=0.3, type="data")
    play((SAMPLE_RATE, low_shelved_test_audio), volume=0.3, type="data")


                                                        #Stretching: 

    stretched_test_audio = time_stretch_fx((SAMPLE_RATE, test_audio), rate_in=2, type="data")
    plot_AB((SAMPLE_RATE, test_audio), (SAMPLE_RATE, stretched_test_audio), mode="overlap", type="data")

    #Play normal and stretched
    play(full_path, volume=0.3)
    play((SAMPLE_RATE, stretched_test_audio), volume=0.3, type="data")

                                                        #Pitch-Shifting: 

    pitch_shifted_test_audio = pitch_shift_fx(test_audio, SAMPLE_RATE, pitch_shift_steps=-2)
    plot_AB((SAMPLE_RATE, test_audio), (SAMPLE_RATE, pitch_shifted_test_audio), mode="overlap", type="data")

    #Play normal and pitch-shifted
    play(full_path, volume=0.3)
    play((SAMPLE_RATE, pitch_shifted_test_audio), volume=0.3, type="data")


                                                        #Masking: 
    
    #Frequency
    freq_masked_test_audio = mask_audio(test_audio, SAMPLE_RATE, mask_type='frequency', max_mask_pct=0.02, plot=True)

    #Play normal and frequency masked
    play(full_path, volume=0.3)
    play((SAMPLE_RATE, freq_masked_test_audio), volume=0.3, type="data")

    #Time
    time_masked_test_audio = mask_audio(test_audio, SAMPLE_RATE, mask_type='time', max_mask_pct=0.02, plot=True)

    #Play normal and frequency masked
    play(full_path, volume=0.3)
    play((SAMPLE_RATE, time_masked_test_audio), volume=0.3, type="data")

def test_compressor():
    
    current_file_path = os.path.abspath(__file__)
    src_dir_path = os.path.dirname(current_file_path)
    project_root = os.path.dirname(src_dir_path)
    data_path = os.path.join(project_root, "data")
    filepath = os.path.join(data_path, "firetruck")
    filename = "sound_303.wav"

    # Join the filepath and filename correctly
    full_path = os.path.join(filepath, filename)

    #Get raw data
    sr, test_audio = wavfile.read(full_path)

    #Get right channel only, as the plugins are all mono
    test_audio_int16 = test_audio[:, 1]

    max_val_int16 = np.max(test_audio_int16)

    print("\n test audio int16 datatype: ", test_audio_int16.dtype)
    print("max value in file: ", max_val_int16)

    #Audio is in 16-bit PCM (WAV), equivalent to np.int16. We want it in float-point for librosa processing
    test_audio_float32_norm = test_audio.astype(np.float32) / np.max(np.abs(test_audio))  # Normalize to [-1.0, 1.0]
    test_audio_float32 = test_audio.astype(np.float32) / np.max(np.abs(test_audio))  

    max_val_float_32 = np.max(test_audio_float32)

    print("\n test audio float32 datatype: ", test_audio_float32.dtype)
    print("max value in file: ", max_val_float_32)

    comp_test_audio_int16 = compressor(test_audio_int16, -17, 3, format="int16", make_up_gain=True)

    plot_AB((SAMPLE_RATE, test_audio_int16), (SAMPLE_RATE, comp_test_audio_int16), type="data")

    #play((SAMPLE_RATE, test_audio_int16), type="data", volume= 0.3)
    #play((SAMPLE_RATE, comp_test_audio_int16), type="data", volume = 0.3)



#Call tests:

#tests()
#test_compressor()
