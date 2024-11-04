import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
import json

"""
This module handles the pre-processing of siren audio data files. It includes functions for 
artificially expanding the dataset and saving it to a dynamic path within the project directory.
The `extract_acoustic_features()` function saves the extracted acoustic features as a JSON file.
When the acoustic data wrapper is invoked, it first checks for the existence of the saved acoustic 
features. If the features are found, they are loaded; otherwise, they are extracted in real-time during
execution. This design optimizes processing time by avoiding repetitive extraction of features, which 
is time-consuming.
"""

#Global variables 
SAMPLE_RATE = 44100
BIT_DEPTH = 16
LABELS = ["ambulance", "firetruck", "traffic"]

#Preprocessing the data

def force_standard_size(dir_path):
    """
    Some files in the vanilla data set have different lengths. This function forces a standard size
    by zero-padding or cutting files that are not 3s in length. 

    Parameters:
    - dir_path: path for the directory of the WAVE files in the vanilla data set

    Returns:
    - Nothing

    Modifies:
    - All files in the data set. Either expands them, cuts them short, or adds a fade-out.  
    """
    for label in LABELS:
        # Create the path to each label folder using the full path
        label_path = os.path.join(dir_path, label)

        # Iterate through all files in the folder
        for file_name in os.listdir(label_path):
            if file_name.endswith('.wav'):  # Check if it's a .wav file
                file_path = os.path.join(label_path, file_name)  # Full path to the file
                sr, audio_data = wavfile.read(file_path)

                target_length = SAMPLE_RATE * 3  # Target length for 3 seconds

                # Check if audio_data is stereo
                if len(audio_data.shape) == 2:
                    # Stereo: create a new array for modified audio
                    modified_audio_data = np.zeros((target_length, audio_data.shape[1]), dtype=np.int16)

                    for channel in range(audio_data.shape[1]):
                        channel_data = audio_data[:, channel]

                        # If larger than 3 seconds, cut to 3 seconds
                        if len(channel_data) > target_length:
                            channel_data = channel_data[:target_length]
                        # If smaller than 3 seconds, zero pad to 3 seconds
                        elif len(channel_data) < target_length:
                            channel_data = np.pad(channel_data, (0, target_length - len(channel_data)), 'constant')

                        # Apply fade-out for the last 0.05 seconds
                        fade_duration = int(SAMPLE_RATE * 0.05)  # 50 ms
                        fade_out = np.linspace(1, 0, fade_duration, endpoint=False)
                        fade_out = fade_out.astype(np.float32)  # Keep as float for multiplication

                        # Apply fade-out only if channel data is longer than fade_duration
                        if len(channel_data) >= fade_duration:
                            channel_data[-fade_duration:] = (channel_data[-fade_duration:] * fade_out).astype(np.int16)

                        # Store modified channel data
                        modified_audio_data[:len(channel_data), channel] = channel_data

                else:
                    # Mono: process as a single channel
                    modified_audio_data = np.zeros((target_length,), dtype=np.int16)

                    if len(audio_data) > target_length:
                        audio_data = audio_data[:target_length]
                    elif len(audio_data) < target_length:
                        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')

                    # Apply fade-out for the last 0.05 seconds
                    fade_duration = int(SAMPLE_RATE * 0.05)  # 50 ms
                    fade_out = np.linspace(1, 0, fade_duration, endpoint=False)
                    fade_out = fade_out.astype(np.float32)  # Keep as float for multiplication

                    # Apply fade-out only if audio is longer than fade_duration
                    if len(audio_data) >= fade_duration:
                        audio_data[-fade_duration:] = (audio_data[-fade_duration:] * fade_out).astype(np.int16)

                    modified_audio_data[:len(audio_data)] = audio_data

                # Overwrite the original file with the modified audio
                wavfile.write(file_path, sr, modified_audio_data.astype(np.int16))  # Save with same sample rate

                print(f"Processed and overwritten {file_name}: length={len(modified_audio_data)} samples")


def vectorized_result(label):
    """
    Return a 3-dimensional unit vector with a 1.0 in the correct index
    corresponding to the label: ambulance -> index 0, firetruck -> index 1, 
    and traffic -> index 2.
    
    Paremeters:
    - label: The proper label of the data, one of {'ambulance', 'firetruck', 'traffic'}

    Returns:
    - vector with 1 in the proper position for the label and 0 at other positions (assume labels are in the previous, alfabetical order)

    Modifies:
    - Nothing
    """
    label_mapping = {'ambulance': 0, 'firetruck': 1, 'traffic': 2}
    e = np.zeros((3, 1))  # Create a vector of zeros with 3 elements
    e[label_mapping[label]] = 1  # Set the correct index to 1
    return e

def expand_data():
    """
    Takes in a data tuple of WAVE files and artificially expands it and writes the expanded audio files in a new directory. 

    Parameters:
    - Nothing 

    Returns:
    - Nothing

    Modifies:
    - Creates a new directory with the artificially expanded data
    """
    import plugins

    def write_file(audio_data, folder_path, filename="output.wav", sample_rate=44100):
        """
        Write normalized audio data to a .wav file, converting the format to int16 if needed.

        Parameters:
        - audio_data : NumPy array containing the audio data (can be float32 or other formats).
        - folder_path: Path to the folder where the .wav file should be saved.
        - filename   : Name of the output .wav file (default is 'output.wav').
        - sample_rate: Sample rate for the WAV file (default is 44100).

        Returns:
        - full_path  : Full path of the saved .wav file.

        Modifies: 
        - folder_path directory: Adds an audio file to such directory
        """
        from scipy.io.wavfile import write

        # Ensure the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Check if audio is in float32 and needs to be converted to int16
        if audio_data.dtype != np.int16:
            # Normalize float32 to the range of int16 (-32768 to 32767)
            audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

        # Define the full file path
        full_path = os.path.join(folder_path, filename)

        # Write the WAV file
        write(full_path, sample_rate, audio_data)

        print(f"Wrote file {filename} to {folder_path}")
        return full_path

    # Create new directory for expanded data and populate it with wav files
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    siren_path = os.path.dirname(src_dir)
    expanded_data_path = os.path.join(siren_path, "artificially_expanded_data")

    # Check if the expanded data directory exists, if not create it
    if not os.path.exists(expanded_data_path):
        os.makedirs(expanded_data_path)

    for label in LABELS:
        # Make directory for artificial data in this label
        label_path = os.path.join(expanded_data_path, label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)

        # Iterate through all wav files in this label present in the vanilla data
        vanilla_label_path = os.path.join(siren_path, "data", label)

        for file_name in os.listdir(vanilla_label_path):
            if file_name.endswith('.wav'):  # Check if it's a .wav file
                # Get full path to the original wav file
                file_full_path = os.path.join(vanilla_label_path, file_name)

                # Create new wav files in expanded data directory under the proper label
                plugins.stereo_to_mono(file_full_path, label_path, mode="sum") # Create mono files

                # Read file data
                sr, audio_int16 = wavfile.read(file_full_path)

                #Normalize current audio
                audio_int16 = (audio_int16/np.max(np.abs(audio_int16)))* 32767

                # Convert to float32 for further processing, if needed
                audio_float_32 = audio_int16.astype(np.float32)

                # Apply compression (in int16)
                comp_file_int16 = plugins.compressor(audio_int16, -18, 4, make_up_gain=True, format="int16")

                # Generate the filename for the limited version
                new_file_name = file_name.split(".")[0] + "_comp.wav"

                # Write it in the current label directory in the expanded data folder
                write_file(comp_file_int16, label_path, filename=new_file_name, sample_rate=sr) #Create compressed files

                #Get low-end boosted files
                lb_audio = plugins.shelving(audio_int16, 18, 200, SAMPLE_RATE, 1/np.sqrt(2), filter_type='low')

                # Generate the filename for the low-boosted version
                new_file_name = file_name.split(".")[0] + "_low_boost.wav"

                 # Write it in the current label directory in the expanded data folder
                write_file(lb_audio, label_path, filename=new_file_name, sample_rate=sr) #Create lb files

                #Get high-end boosted files
                hb_audio = plugins.shelving(audio_int16, 18, 3000, SAMPLE_RATE, 1/np.sqrt(2), filter_type='high')

                # Generate the filename for the high-boosted version
                new_file_name = file_name.split(".")[0] + "_high_boost.wav"

                # Write it in the current label directory in the expanded data folder
                write_file(hb_audio, label_path, filename=new_file_name, sample_rate=sr) #Create hb files

                #Get low-end cut files
                lc_audio = plugins.shelving(audio_int16, -20, 200, SAMPLE_RATE, 1/np.sqrt(2), filter_type='low')

                # Generate the filename for the low-boosted version
                new_file_name = file_name.split(".")[0] + "_low_cut.wav"

                 # Write it in the current label directory in the expanded data folder
                write_file(lc_audio, label_path, filename=new_file_name, sample_rate=sr) #Create lc files

                #Get high-end boosted files
                hc_audio = plugins.shelving(audio_int16, -20, 3000, SAMPLE_RATE, 1/np.sqrt(2), filter_type='high')

                # Generate the filename for the high-cut version
                new_file_name = file_name.split(".")[0] + "_high_cut.wav"

                # Write it in the current label directory in the expanded data folder
                write_file(hc_audio, label_path, filename=new_file_name, sample_rate=sr) #Create hc files

                #Other possibilities: 

                #Get pitched-up file from plugins
                #pitched_up_audio = plugins.pitch_shift_fx(audio_float_32, SAMPLE_RATE, 3)

                # Generate the filename for the pitch-up version
                #new_file_name = file_name.split(".")[0] + "_pitched_up.wav"

                # Write it in the current label directory in the expanded data folder
                #write_file(pitched_up_audio, label_path, filename=new_file_name, sample_rate=sr) #Create pitched up files

                #Get pitched-down file from plugins
                #pitched_down_audio = plugins.pitch_shift_fx(audio_float_32, SAMPLE_RATE, -3)

                # Generate the filename for the pitch-up version
                #new_file_name = file_name.split(".")[0] + "_pitched_down.wav"

                # Write it in the current label directory in the expanded data folder
                #write_file(pitched_down_audio, label_path, filename=new_file_name, sample_rate=sr) #Create pitched up files """


    print(f"\n Finished generating new files and adding them to {expanded_data_path}.")


def acoustic_data_wrapper(mode="standard"):
    """ 
    Shuffle data, then load training, validation, and test data with reshaping. 
    This wrapper is similar to data_wrapper, but instead of loading spectrograms,
    it loads acoustic features extracted from the .wav files using extract_acoustic_features. 

    Parameters:
    - mode (str): The mode of operation (e.g., "standard" or other modes). 

    Returns:
    - data: Loaded and reshaped acoustic features based on the specified mode.

    Modifies:
    - Nothing.
    """

    print(f"Loading data in Mode {mode}")

    # Get the project root directory dynamically based on the location of this script
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)  # Get directory of this script (src)
    project_root = os.path.dirname(src_dir)  # Go up one level to project root (Siren_Classifier)

    if mode == "standard":
        # Load the data once at the beginning
        data_folder = os.path.join(project_root, 'data')  # Adjusted to dynamic path

        # Initialize the acoustic data tuples
        feature_tuples_path = os.path.join(project_root, "acoustic_features", "acoustic_features_standard.json")
        
        if not os.path.exists(feature_tuples_path):
            acoustic_data_tuples = extract_acoustic_features(data_folder, json_file_name="acoustic_features_standard")
        else:
            print("Acoustic features already exist, loading it...")
            acoustic_data_tuples = load_acoustic_features(feature_tuples_path)

        # Shuffle the pre-loaded data
        random.shuffle(acoustic_data_tuples)

        # Reshape the data to match the format used for classification
        reshaped_data = []
        for feature_vector, label in acoustic_data_tuples:
            flattened_features = np.hstack(feature_vector).reshape(-1, 1)
            vectorized_label = vectorized_result(label)
            reshaped_data.append((flattened_features, vectorized_label))

        # Split into training, validation, and test sets
        training_data = reshaped_data[:400]
        validation_data = reshaped_data[400:500]
        test_data = reshaped_data[500:600]

        return (training_data, validation_data, test_data)

    elif mode == "expand":
        
        #Check if the artificially expanded data exists
        expanded_data_folder = os.path.join(project_root, 'artificially_expanded_data')
        if not expanded_data_folder:
            expand_data() #Expand data and generate the directory if not
        mode = 'expanded' #Proceed in expanded mode

    elif mode == "expanded":

        #Get expanded data tuples
        expanded_data_folder = os.path.join(project_root, 'artificially_expanded_data')
        expanded_feature_tuples_path = os.path.join(project_root, "acoustic_features", "acoustic_features_expanded.json")

        if not os.path.exists(expanded_feature_tuples_path):
            expanded_acoustic_data_tuples = extract_acoustic_features(expanded_data_folder, json_file_name='acoustic_features_expanded')
        else:
            print("Acoustic features already exist, loading it...")
            expanded_acoustic_data_tuples = load_acoustic_features(expanded_feature_tuples_path)

        #Shuffle data tuples
        random.shuffle(expanded_acoustic_data_tuples)

        #Reshape data and vectorize labels 
        reshaped_data = []
        for feature_vector, label in expanded_acoustic_data_tuples:
            flattened_features = np.hstack(feature_vector).reshape(-1, 1)
            vectorized_label = vectorized_result(label)
            reshaped_data.append((flattened_features, vectorized_label))

        #Distribute data tuples amopng the return variables
        length = len(reshaped_data)
        train_size = int(np.floor(0.75 * length))
        val_size = int(np.floor((length - train_size) / 2))

        training_data = reshaped_data[:train_size]
        validation_data = reshaped_data[train_size:train_size + val_size]
        test_data = reshaped_data[train_size + val_size:]

        return (training_data, validation_data, test_data)

def extract_acoustic_features(data_directory, json_file_name='acoustic_features.json'):
    """
    Extracts acoustic features from all .wav files in each label folder inside 'data_directory'. 
    Features extracted:
    - Mel-frequency cepstral coefficients (MFCCs) [Shape: (13, T)]
    - Zero Crossing Rate (ZCR) [Shape: Scalar]
    - Energy ratios across 15 frequency bands [Shape: (14,)]

    Parameters:
    - data_directory (str): Where the audio files whose features will be extracted are located
    - json_file_name (str): The name of the JSON file generated with the acoustic features of each file in 
    data_directory

    Returns:
    - List of tuples: Each tuple is (feature_vector, label)

    Modifies:
    - project root directory: Creates a new directory called 'acoustic_features' in the project root. 
    """
    print("Extracting acoustic features...")

    def energy_per_frequency_band(audio_data, sr, num_bands=15):
        """
        A function to calculate the energy ratios of num_bands in the frequency domain
        """
        fft_result = np.fft.fft(audio_data)
        fft_magnitude = np.abs(fft_result)
        band_size = len(fft_magnitude) // num_bands
        band_energies = [np.sum(fft_magnitude[i * band_size: (i + 1) * band_size] ** 2) for i in range(num_bands)]
        energy_ratios = [band_energies[i] / band_energies[0] for i in range(1, num_bands)]
        return np.array(energy_ratios)

    def zero_crossing_rate(audio_data):
        """
        A function to calculate the zero crossing rate of an audio file
        """
        return 0.5 * np.sum(np.abs(np.diff(np.sign(audio_data))))

    def sum_to_mono(audio_data):
        """
        A function to sum stereo files to mono

        Note: this file returns a mono file, unlike stereo_to_mono() in plugins.py, 
        which by default writes new files to a given directory and returns nothing. 
        """
        if audio_data.ndim == 1:
            return audio_data
        mono_audio = np.sum(audio_data, axis=1)
        max_val = np.max(np.abs(mono_audio))
        if max_val > 0:
            mono_audio = mono_audio * (32767 / max_val)
        return np.clip(mono_audio, -32768, 32767).astype(np.int16)

    #Create list fot the feature_tuples
    feature_tuples = []

    length_freq_dict = {}

    #Loop through all labels in this data_directory
    for label in LABELS:
        label_path = os.path.join(data_directory, label)
        if not os.path.isdir(label_path):
            continue
        #Loop through all wav files in this label directory
        for file_name in os.listdir(label_path):
            if file_name.endswith('.wav'):
                #Get audio_data
                file_path = os.path.join(label_path, file_name)
                sr, audio_data = wavfile.read(file_path)

                #Sum audio_data to mono and normalize it 
                audio_data = sum_to_mono(audio_data)

                 # Count the length of audio_data
                length = len(audio_data)
                if length in length_freq_dict:
                    length_freq_dict[length] += 1
                else:
                    length_freq_dict[length] = 1

                #Get zero crossings
                zcr = zero_crossing_rate(audio_data)

                #Get energy bands
                energy_ratios = energy_per_frequency_band(audio_data, sr)

                                # Get MFCCs (convert to float32 to work with librosa)
                audio_data_float32 = audio_data.astype(np.float32)  # Convert to float
                audio_data_float32 /= np.max(np.abs(audio_data_float32))  # Normalize to [-1, 1]

                mfccs = librosa.feature.mfcc(y=audio_data_float32, sr=sr) 

                # Flatten the MFCCs across time frames and pack them into the feature vector
                feature_vector = [zcr] + energy_ratios.tolist() + mfccs.flatten().tolist()

                feature_tuples.append((feature_vector, label))

    # Get the project root directory dynamically based on the location of this script
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)  # Get directory of this script (src)
    project_root = os.path.dirname(src_dir)  # Go up one level to project root (Siren_Classifier)
    json_path = os.path.join(project_root, "acoustic_features") # Set path to acostic_features

    #If the directory does not exist, create it
    if not os.path.exists(json_path):
        os.makedirs(json_path)


    json_file_path = os.path.join(json_path, f"{json_file_name}.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(feature_tuples, json_file, indent=4)

    print(length_freq_dict)
    print(f"Extracted Acoustic features and saved to JSON in file {json_file_path}")
    return feature_tuples

def load_acoustic_features(json_file_path):
    """
    Loads acoustic features from a JSON file.

    Parameters:
    - json_file_path: The file path to the extracted acoustic features

    Returns:
    - List of tuples: Each tuple is (feature_vector, label).

    Modifies:
    - Nothing

    """
    with open(json_file_path, 'r') as f:
        feature_tuples = json.load(f)
    print("Loaded Acoustic features.")
    return feature_tuples


def data_visualizer(data_tuples, index):
    """
    Visualizes the acoustic features along with its label from the data tuples.

    Parameters:
    - data_tuples: List of tuples containing (feature_vector, label).
    - index: Index of the data point to visualize.

    Returns:
    - Nothing

    Modifies:
    - Nothing
    """
    # Get the feature vector and label from the list at the given index
    feature_vector, label = data_tuples[index]

    # Extract features
    zcr = float(feature_vector[0])  # Ensure ZCR is a scalar
    energies = feature_vector[1:15]  # Energy band ratios
    mfccs = feature_vector[15:]  # Remaining features (MFCCs)

    # Ensure energies is a 1D array
    energies = np.asarray(energies).flatten()

    # Plot for energies (energy band ratios)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(energies)), energies, color='skyblue', alpha=0.7, label='Energy Band Ratios')
    plt.ylabel('Energy Value')
    plt.xlabel('Energy Band Index')
    plt.title(f'Energy Band Ratios - Label: {label.flatten()}')
    plt.grid()
    
    # Add text for Zero Crossings below the plot
    plt.text(0, -0.05 * max(energies), f'Zero Crossings: {zcr:.2f}', fontsize=10, ha='left')  # Adjust y-position as needed

    # Plot for MFCCs in a new figure
    plt.figure(figsize=(10, 6))
    plt.title(f'MFCCs - Label: {label.flatten()}')
    plt.plot(mfccs, marker='o', color='orange', label='MFCCs')
    plt.ylabel('MFCC Value')
    plt.xlabel('MFCC Index')
    plt.grid()

    # Display both plots
    plt.show()
    return
