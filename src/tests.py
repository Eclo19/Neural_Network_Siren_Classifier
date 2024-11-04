import siren_loader
import my_net as nw
import os

"""
This script includes tests for neural network training and data handling specific to the Siren Classifier project. 
The primary function, `test_train_NN()`, enables hyperparameter experimentation and monitors network performance, 
saving the best model based on validation accuracy at each epoch. Additional functions test data loading, feature 
extraction, and MFCC calculation for audio data, ensuring proper preprocessing and debugging support. Use this 
script to experiment with training and hyperparameter tunning.
"""

def test_siren_loader(mode="standard"):
    """Tests loading data and visualizing samples using specified mode.
    
    Args:
        mode (str): Specifies whether to use 'standard' or 'expanded' data.
    """

    if mode == "standard":
        print("                                      Testing standard acoustic data loading...")
        training_data, validation_data, test_data = siren_loader.acoustic_data_wrapper(mode="standard")
    elif mode == "expanded":
        print("                                      Testing expanded acoustic data loading...")
        training_data, validation_data, test_data = siren_loader.acoustic_data_wrapper(mode="expanded")
    elif mode == "expand":
        print("                                      Testing expanded acoustic data loading...")
        training_data, validation_data, test_data = siren_loader.acoustic_data_wrapper(mode="expand")
    else:
        raise ValueError(f"Invalid mode {mode}! Must be 'standard', 'expanded', or 'expand.")

    print("                                      Loaded data")

    # Print out the details of the loaded data
    print("\nData Loaded:")
    print(f"Training Data Points: {len(training_data)}")
    print(f"Validation Data Points: {len(validation_data)}")
    print(f"Test Data Points: {len(test_data)}\n")

    # Print samples of training data
    print("\nSample Training Data (First 5):")
    for i in range(min(5, len(training_data))):
        print(f"Data Shape: {training_data[i][0].shape}, Label: {training_data[i][1].flatten()}")

    # Plot an example data
    siren_loader.data_visualizer(training_data, 27)

    total_num_data = len(training_data) + len(validation_data) + len(test_data)
    print("\nTotal number of datapoints: ", total_num_data)

   
def test_mfcc():
    audio_path = '/Users/ericoliviera/Desktop/My_Repositories/Siren_Classifier/data/ambulance/sound_3.wav'
    from scipy.io import wavfile
    import librosa
    import numpy as np

    # Read the WAV file
    sr, audio_data = wavfile.read(audio_path)

    # Sum audio_data's L and R channels to get a mono file
    if audio_data.ndim == 2:  # Check if the audio is stereo
        audio_data = np.sum(audio_data, axis=1)  # Sum across the channels to get mono

    # Convert to float32 and normalize
    audio_data_float32 = audio_data.astype(np.float32)  # Convert to float
    audio_data_float32 /= np.max(np.abs(audio_data_float32))  # Normalize to [-1, 1]

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data_float32, sr=sr)

    print(f"Shape of mfccs: {mfccs.shape}")
    print(mfccs)

     
def test_feature_list():
    # Extract data tuples from standard data:
    data_tuples = siren_loader.extract_acoustic_features('/Users/ericoliviera/Desktop/My_Repositories/Siren_Classifier/data')

    # Load the data tuples
    #data_tuples = siren_loader.load_acoustic_features('/Users/ericoliviera/Desktop/My_Repositories/Siren_Classifier/acoustic_features/acoustic_features_standard.json')


    # Ensure data_tuples is loaded correctly
    if not data_tuples:
        print("Error: No data loaded.")
        return
    

    # Initialize structures for debugging information
    labels = set()
    data_lengths = {}  # Dictionary to store each data length by index
    tuple_sizes = []  # List to store lengths of each tuple
    length_frequency = {}  # Dictionary to count occurrences of each unique length

    # Parse each tuple in data_tuples
    for idx, (data, label) in enumerate(data_tuples):
        labels.add(label)  # Collect unique labels
        length = len(data)  # Get the length of the current data vector
        data_lengths[idx] = length  # Store the length of each data vector by index
        tuple_sizes.append(length)  # Add length to a list for quick size inspection
        
        # Count the frequency of each unique length
        if length in length_frequency:
            length_frequency[length] += 1
        else:
            length_frequency[length] = 1

    # Output debug information
    print("Debugging Information:")
    print(f"Total number of data tuples: {len(data_tuples)}")
    print(f"Unique labels found: {labels}")
    print(f"Overall shape of data_tuples: {len(data_tuples)} tuples of varying data length.")
    print(f"List of unique data lengths in tuples: {set(tuple_sizes)}")  # Unique lengths for quick inspection

    # Additional statistics
    min_length = min(tuple_sizes)
    max_length = max(tuple_sizes)
    print(f"Minimum feature vector length: {min_length}")
    print(f"Maximum feature vector length: {max_length}")

    # Print frequency of each unique length
    print("Frequency of each feature vector length:")
    for length, count in length_frequency.items():
        print(f"Length {length}: {count} occurrences")

    # Optional: Check if all tuples have consistent data lengths
    if len(set(tuple_sizes)) > 1:
        print("Warning: Inconsistent data lengths found among feature vectors.")
    else:
        print("All data tuples have consistent feature vector lengths.")

def test_trained_NN(path, softmax=False):
    """
    Load a saved neural network and test data from `path` and evaluate its accuracy on the test data
    
    Args:
        path (str): Path to the saved network. 
        softmax (bool): If true. dispplays the errors and shows the probability distributions of the possible labels in a mistake
    
    """
    
    print("Importing Network...")
    # Use the `load` function to retrieve the network and test data
    net, test_data = nw.load(path)
    if test_data == None:
        raise ValueError("No test data provided. `test_trained_NN()` requires a network saved with its training data.)")
    print("Network Imported")

    # Display mistakes in accuracy
    net.showMistakes = True

    print(f"Test data length: {len(test_data)}")

    # Evaluate the network on the test data
    accuracy = net.accuracy(test_data, softmax_layer=softmax)
    print(f"Accuracy on test data: {accuracy} / {len(test_data)} ({(100 * accuracy / len(test_data)):.3f}%)")


def test_train_NN(mode="standard"):
    """Trains and tests the neural network using either spectrogram or acoustic data.
    
    Args:
        mode (str): Specifies whether to use 'standard' or 'expanded' data.
    """

    test_data = None
    training_data = None
    validation_data = None

    # Load data based on the mode
    if mode == "standard":
        print("                                      Testing NN with standard acoustic data...")
        training_data, validation_data, test_data = siren_loader.acoustic_data_wrapper(mode="standard")
    elif mode == "expanded":
        print("                                      Testing NN with expanded acoustic data...")
        training_data, validation_data, test_data = siren_loader.acoustic_data_wrapper(mode="expanded")
    else:
        raise ValueError("Invalid mode! Must be 'standard' or 'expanded'.")
    

    print("                                      Loaded data")


    if mode == 'standard':

        # Retrieve a test sample to determine the input shape
        test_sample = test_data[25]
        sample, true_label = test_sample
        print("sample data shape:", sample.shape)

        # Define the network structure with input size determined dynamically
        sizes = [sample.shape[0], 500, 30, 3]  # The first layer size matches the input data dimensions

        # Initialize the network with the computed layer sizes
        net = nw.Network(sizes)

        # Enter test_data into Network instance so it is saved with the network
        net.test_data = test_data

        # Set training parameters 
        epochs = 50
        m = 10
        eta = 0.001
        l = 0.01
        p = 15

        print("                                      Starting Training with standard data...")

        # Train the network using stochastic gradient descent with stabdard data
        net.SGD(training_data=training_data, 
                epochs=epochs,
                mini_batch_size=m,
                eta=eta, 
                evaluation_data=validation_data, 
                monitor_evaluation_accuracy=True, 
                monitor_evaluation_cost=True, 
                monitor_training_accuracy=True, 
                monitor_training_cost=True,
                lmbda=l,
                patience=p, 
                variable_eta=True, 
                save_network=True)

        print("                                      Training complete")

    if mode == 'expanded':

        # Retrieve a test sample to determine the input shape
        test_sample = test_data[25]
        sample, true_label = test_sample
        print("sample data shape:", sample.shape)

        # Define the network structure with input size determined dynamically
        sizes = [sample.shape[0], 500, 10, 3]  # The first layer size matches the input data dimensions [in, 500, 10, out] curr best of 94.22%

        # Initialize the network with the computed layer sizes
        net = nw.Network(sizes)

        # Enter test_data into Network instance so it is saved with the network
        net.test_data = test_data

        # Set training parameters 
        epochs = 200
        m = 40
        eta = 0.002
        l = 0.1
        p = 8
        train_n = 1200
        eval_n = 250

        print("                                      Starting Training with expanded data...")

        # Train the network using stochastic gradient descent with expanded data
        net.SGD(training_data=training_data, 
                epochs=epochs,
                mini_batch_size=m,
                eta=eta, 
                evaluation_data=validation_data, 
                monitor_evaluation_accuracy=True, 
                monitor_evaluation_cost=False, 
                monitor_training_accuracy=True, 
                monitor_training_cost=False,
                lmbda=l, 
                patience=p, 
                variable_eta=True, 
                save_network=True)

        print("                                      Training complete")

# To experiment with Training, call the following functin with the mode and data set accodrding to your intentions
#test_train_NN(mode="standard")

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)  # Get directory of this script (src)
project_root = os.path.dirname(src_dir)  # Go up one level to project root (Siren_Classifier)

#Check the accuracy on the test data with high-performing networks that I saved using test_train_NN(), from tests.py

#Network with the best accuracy on validation data
best_ac_validation_path = os.path.join(project_root, 'Best_NNs', '01_11_11_02_best_network_ac_of_96.22.json')

#Network with the best accuracy on test data
best_ac_test_path = os.path.join(project_root, 'Best_NNs', '01_11_15_23_best_network_ac_of_94.44.json')

#Check the accuracy on the test data of a given network:
test_trained_NN (best_ac_test_path, softmax=True) # Best test_data accuracy

#test_siren_loader(mode="standard", data_type="acoustic")
