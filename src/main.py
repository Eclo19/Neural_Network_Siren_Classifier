
import siren_loader as loader
import tests
import os

"""
This project implements a classifier for distinguishing between different types of emergency siren sounds. The dataset includes 
three labels: 'ambulance' (0), 'firetruck' (1), and 'traffic' (2). This script checks that all relevant data is properly 
downloaded and re-generates it if not found. The main() function runs a test on one of the best-performing networks saved during 
training experiments. For those interested in experimenting with training and hyperparameter tuning, refer to the test_train_NN() 
function inside tests.py. 

Inspired by Michael Nielsen's book "Neural Networks and Deep Learning" (available for free at http://neuralnetworksanddeeplearning.com/), 
I embarked on this project to tackle a classification challenge similar to the MNIST dataset. As an audio enthusiast, I sought an 
audio classification project with real-world applications. I discovered the original dataset on Kaggle, and I believe this problem 
has practical potential (e.g., in smart car technology). If deployed, this classifier could predict whether an audio stream contains 
a siren and identify the type of siren.

To create this audio classifier, I am extracting acoustic features from the datasets and training a custom-built, fully connected 
neural network to recognize patterns in these features. Developing a robust network requires substantial experimentation, and this 
project serves as an educational resource for myself and others interested in audio machine learning. This code is not optimized 
for production, as I opted to hard-code the network to better understand its inner workings.

**Note**: If you downloaded the original dataset linked in the README file (https://www.kaggle.com/datasets/vishnu0399/emergency-vehicle-siren-sounds), 
call `force_standard_size()` to ensure all files are uniform in length, as a few samples deviate in duration. If you obtained the 'data' folder 
from my GitHub, this function has already been applied, and there should be no issues with file shapes and sizes.
"""


# Global variables
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)  # Get directory of this script (src)
project_root = os.path.dirname(src_dir)  # Go up one level to project root (Siren_Classifier)
data_dir = os.path.join(project_root, 'data') # Get path to original vanilla data

def check_and_set():

    """
    This functino checks if all the supporting data folders exists and creates them if the folders are not found
    within the project directory. 
    """

    print("\nInitiating checks...")

    #Create artificially expanded data directory if it does not exist
    expanded_data_dir = os.path.join(project_root, 'artificially_expanded_data')

    if not os.path.exists(expanded_data_dir):
        print("Expanding vanilla data set...")
        loader.expand_data()

    #Extract acoustic features for both vanilla and artificially expanded data
    acoustic_features_dir = os.path.join(project_root, "acoustic_features")

    if not os.path.exists(acoustic_features_dir):
        print("Extracting acoustic features from vanilla data_set..")
        loader.extract_acoustic_features(data_directory=data_dir, json_file_name='acoustic_features_standard')
        print("Done.")

        print("Extracting acoustic features from expanded data_set...")
        loader.extract_acoustic_features(data_directory=expanded_data_dir, json_file_name='acoustic_features_expanded')
        print("Done.")

    print("Checks completed succsesfully.\n")


def force_standard_size():
        
    """
    This function alters the data in the original vanilla data set from Kaggle and ensures all files are of equal length. 
    """

    print(f"Forcing all files in '{data_dir}' to be of length 3s...")
    loader.force_standard_size(data_dir)
    print("Done.")
    return
    

def main():
    # Check and set up necessary directories and files
    check_and_set()

    #Test one of the most accurate NN's

    #Network with the best accuracy on validation data
    best_ac_validation_path = os.path.join(project_root, 'Best_NNs', '01_11_11_02_best_network_ac_of_96.22.json')

    #Network with the best accuracy on test data
    best_ac_test_path = os.path.join(project_root, 'Best_NNs', '01_11_15_23_best_network_ac_of_94.44.json')

    tests.test_trained_NN(best_ac_test_path)


if __name__ == "__main__":
    main()







