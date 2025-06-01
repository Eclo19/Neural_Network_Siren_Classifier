# Neural Network Classifier For Emergency Siren Sounds 

## Motivation & Overview

This repository contains a personal project that creates a Neural Networks for audio classification of urban emergency siren sounds. My main goal with this project is to strengthen my understanding of Neural Networks while creating relevant material for my portfolio. With an interest in audio applications, I chose to work on a classification system for urban siren sounds, which could have practical applications, such as alert systems in smart cars.

This project creates an audio classifier for siren sounds: it classifies incoming data into one of the labels (types of siren sounds) it was trained to recognize. The code is not at all optimized, as it was based on Michael Nielsen's ([GitHub](https://github.com/mnielsen)) educationally-focused neural network implementations presented in his wonderful book "Neural Networks and Deep Learning" ([neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/chap1.html)), where he introduces Neural Networks while tackling the MNIST dataset.

I implemented several of the features proposed as problems in the book as well as some original ones to properly handle my data and manage when and how to save the network. I chose not to use standard deep learning libraries such as Pytorch or TensorFlow because I wanted to experience coding the network from scratch and controlling every step of the process. 

If you would like to use this to practice training neural networks and hyperparameter tuning, run `test_train_NN()` in `test.py` with either the vanilla (standard) or the artificially expanded dataset. There is also room for experimentation inside the `expand_data()` and the `extract_acoustic_features()` functions in `siren_loader.py`. 

**-- I achieved a maximum accuracy on data not seen during training of 96.67% --** 

### -- If you have any questions about this project or are interested in connecting please reach out to me! I have contact information in my profile. -- 

## Data

### Original Dataset

The vanilla training data was sourced from [Kaggle](https://www.kaggle.com/datasets/vishnu0399/emergency-vehicle-siren-sounds). This dataset consists of `.wav` files with a sample rate of 44.1 kHz and a bit depth of 16. While most files in the dataset are stereo and 3 seconds long, some inconsistencies exist, with variations in length and channels, which are handled by the code (explained below).

The dataset includes three labels:
- `ambulance`
- `firetruck`
- `traffic`

The idea is to have a network that performs well in distinguishing regular traffic sounds from emergency sirens and to also be able to identify which type of siren it is listening to.

### Data Augmentation

A significant part of this project involved creating algorithms for data augmentation to artificially expand the dataset. This process is managed through functions and plugins I developed in `plugins.py`. These audio plugins apply transformations to generate slightly altered versions of the original sounds, thereby increasing the diversity of the training data.

### Note on Large Files

This project deals with significant amounts of data, which exceeded my GitHub storage capacity. Therefore, the `main.py` script checks for and generates necessary files (like `artificially_expanded_data` and `acoustic_features`). If you prefer to download these files instead of generating them—which is considerably faster—they are available in my [Google Drive](https://drive.google.com/drive/folders/1UQZnr5Qf16cyHA6aKMKtSrzYyjGPUXId?usp=sharing). In this repository, you will find only the vanilla dataset and the highest-performing networks from my training, stored using GitHub's LFS.

#### **What This Means for You**:

- **Cloning the Repository**: When you clone the repository, Git LFS will automatically download the large files. 

  To install Git LFS and pull large files:

  1. Visit the [GitHub Docs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage) for detailed instructions.
     
  2. With Git LFS installed, run the following commands:
  
     ```bash
     git lfs install  # Installs Git LFS
     git clone <this-repository-url>  # Clones the repository
     git lfs pull  # Downloads large files
     ```

- **Alternative Without Git LFS**:  
  If you prefer not to install Git LFS:
  
  - Ensure you have the vanilla data (in the `data` folder) or download the original dataset [here](https://www.kaggle.com/datasets/vishnu0399/emergency-vehicle-siren-sounds). Note that using raw data from Kaggle may cause errors in the code. To fix this, run the `force_standard_size()` function in `main.py`, which will modify the files in place for proper formatting. Executing `main.py` will generate all necessary files for this project and store them in the project directory using the `os` Python library.

* Here are all the files you should have in the project root (files will be generated by `main.py` if missing):

 - `data`: Stores the vanilla dataset 
 - `artificially_expanded_data`: Contains augmented audio files generated through data augmentation algorithms.
 - `acoustic_features`: Includes 2 JSON files with the acoustic features of the standard and expanded datasets.
 - `Best_NNs`: Holds 2 JSON files detailing my highest-performing neural networks.
 - `src`: Contains the Python scripts used in the project.

## Workflow

### Ensuring The Data Is Uniform
This project requires initial preprocessing to ensure uniformity in the data. Specifically, the audio files need to have the same length so that extracted acoustic features have consistent shapes (essential for input to the neural network). To standardize the dataset, I wrote the `force_standard_size()` function in `siren_loader.py`, which enforces a uniform file length of 3s and ads a fade-out to all files. Running this function will modify the data in place, so it's not necessary to run it if you're using the pre-processed `data` folder in this repository, as it was already modified by the function.


### Expanding The Dataset
The original dataset contains only 600 samples, which is insufficient for fully training a neural network of the type I defined. To address this, a data expansion step is implemented via the `expand_data()` function in `siren_loader.py`. This function relies on custom audio plugins in `plugins.py` to augment the dataset, generating variations of the original sounds. You can call this function directly or through the `acoustic_data_wrapper()` function with mode='expand'.


### Extracting Acoustic Features
Since the audio files are high-dimensional (e.g., $$44100 \text{ Hz} \times 3 \text{ s} = 132300$$), the program reduces this complexity by extracting acoustic features with the `extract_acoustic_features()` function. This function takes `data_directory` and `json_file_name` as input parameters, processing all audio files in the specified data directory and saving extracted features to a JSON file under the "json_file_name" directory.

The feature extraction process:
- **Stereo to Mono Conversion:** Files are checked for stereo channels, and if stereo, they are summed to mono.
- **Feature Types:** 
  - **Global Features:** These include the zero-crossing rate and energy band ratios, calculated over the entire sample.
  - **Time-Framed Features:** Mel Frequency Cepstral Coefficients (MFCCs) are extracted to capture time-dependent aspects.

While the global features are custom-coded, MFCCs are computed using the `librosa` implementation with default settings. By storing features in JSON format, the feature extraction only needs to occur once, making data loading much faster in subsequent steps. 

### Loading The Data

The `acoustic_data_wrapper()` prepares and manages batches of data. It has three operational modes: `standard`, `expand`, and `expanded`. These modes offer flexibility depending on whether we’re using the vanilla dataset, need to generate expanded data on-the-fly, or simply want to load the pre-expanded dataset.

1. **Data Verification**: When loading data, `acoustic_data_wrapper()` checks for pre-processed acoustic feature files. If these files are missing, it will automatically call `extract_acoustic_features()` to generate and save these features for future use.

2. **Data Wrapping**: After verifying or generating the feature files, this function shuffles and segments the data into three primary sets:
   - **Training Data (75%)**: Used to train the model and learn the patterns in the siren sounds.
   - **Validation Data (12.5%)**: Evaluates model performance during training, helping prevent overfitting.
   - **Test Data (12.5%)**: Measures model accuracy on data it hasn’t seen before, providing a final evaluation metric.

3. **Data Output**: The data is returned in a format optimized for neural network training: each element is a tuple consisting of a flatten acoustic feature vector and a flatten vectorized label.

This structured data flow enables efficient training, validation, and testing of the network, ensuring that each dataset is balanced and consistent with the expected input and output shapes for the model.

### Training The Network

Once the data is prepared, the model training begins with the `SGD()` (Stochastic Gradient Descent) method. This function initiates training based on the given hyperparameters, training data, and optional validation data. 

1. **Training Process**: During training, `SGD()` runs through a specified number of epochs, adjusting the weights and biases in the network to minimize prediction errors. Validation data (if provided) is used to check the model’s accuracy at each epoch, preventing overfitting and providing insights into how well the model generalizes beyond the training data.

2. **Hyperparameter Tuning**: The `SGD()` method includes multiple parameters and flags that control the learning rate, batch size, number of epochs, and regularization terms, allowing experimentation with various configurations to optimize model performance. 

3. **Saving the Network**:If flagged and (optional) provided with test data, the `SGD()` method saves the best performing network on evaluation data and that network's test data as a JSON file. Training runs and performance metrics are recorded in `test_train_NN()`, where experimentation with hyperparameters is performed and the most accurate network versions are preserved.

### 6. Testing

Throughout the project, tests were created to ensure code functionality and verify model accuracy. Testing functions span from evaluating individual components (like audio feature extraction) to assessing the final network performance.

1. **Feature Testing**: Key functions are tested within their respective files, like `plugins.py`, where audio plugins were tested by listening to and visually inspecting outputs. For core functions in `siren_loader.py`, targeted tests such as `test_siren_loader()`, `test_mfcc()`, and `test_feature_list()` in `tests.py` verify that features are correctly processed.

2. **Network Evaluation**: I found it more convenient to pack training into a test function, so to train a new network, call `test_train_NN()`. In this function, a network is created with the user's choice of hyper-parameters and training and validation data, and there are several `print` statements to monitor performance. To test a trained network, call `test_trained_NN()`, which loads saved networks and runs them against their test data to evaluate accuracy on unseen samples. This provides a clear indication of model performance in real-world conditions, confirming that the network can classify sounds accurately outside of the training dataset. For training, use 'test_train_NN()' 


## Code

The Python code contains document strings for each script and fuction that detail how they work and what they do. Below is a generic view of each script, with an emphasis on `my_net.py`, where I define a neural network class `Network` and some helpful functinos/classes. 

### main.py

This function packs up the workflow presented above. For efficiency, I added to the repository all the files the code generates, but this script checks for their existance and re-generates them in the proper directories if they are not found. If this project is ever to be deployed, it will look quite similar to this main script, where a high-performing network is loaded and fed unseen data, but without a label. The `main()` function in the script is simply calling `test_trained_trained_NN()` on a saved network and its test data. 

**Note** If using the original dataset from Kaggle, run the `force_standard_size()` function on the dataset to ensure all files have the same length. 

### my_net.py
Here I implemented my own fully-connected neural network, the heart of this classifier, based on Nielsen's code. This implementation supports as many layers as the user wants and checks that the input and output layers have the same shape as the training data and the vectorized label, respectively. Here are some expressive features defined in this script:

1. **Cost Functions:** There are 2 cost functions: The Quadratic Cost Function and the Cross-Entropy Cost Function. These are implemented as classes with static methods for the function definitions and the respective deltas (error in the final layer with respect to the weighted sum input 'z' of the output layer). I focused on training networks with the Cross-Entropy lost function, as the performance was noticeably better.
2. **Weight Initialization:** The weight initialization follows a Gaussian distribution with mean 0 and standard deviation 1 over the square root of the number of weights connecting to the same neuron. This is essential to get a Gaussian distribution with a smaller standard deviation and therefore a more expressive "peak", which leads to smaller initial weights being sampled from the whole distribution, avoiding neuron saturation.
3. **Activation Function:** All neurons are sigmoid, as this is the only activation function I am using in the network. The sigmoid function and its derivative are defined at the end of the script with clips to avoid overflow.
4. **L2-Regularization:** In order to contain overfitting in the training data, I implemented L2 regularization to penalize large weights. The extra hyperparameter $$\lambda $$ provides more flexibility and leads to higher accuracies when fine-tuned. 
5. **Training Methods:** Training is performed by calling the `SGD()` function, which performs Stochastic Gradient Descent. The method shuffles the training data, creates mini batches, and calls the `update_mini_batch()` method in each mini batch, which updates the weights and biases of the network based on the average gradient vector of each mini-batch. This method does that by calling `backprop()`, which implements the backpropagation algorithm for calculating the gradients of each weight and bias in the mini batch, then adding up all of the gradient vectors for each of the weights and biases in the mini batch. It then follows the L2-regularization update rules:

   - **Weight Update Rule:**
   
      $$w' = \left(1 - \frac{\eta \cdot \lambda}{n}\right) \cdot w - \left(\frac{\eta}{m}\right) \cdot \nabla_w$$
   
   Where:
   - \( w' \) is the updated weight.
   - \( w \) is the current weight.
   - \( $$\eta$$ \) is the learning rate.
   - \( $$\lambda$$ \) is the regularization strength.
   - \( n \) is the total number of training examples.
   - \( m \) is the number of examples in the mini-batch.
   - \( $$\nabla_w$$ \) is the gradient generated by a mini batch of the cost function with respect to the weights.
  
   - **Bias Update Rule:**
  
      $$b' = b - \left(\frac{\eta}{m}\right) \cdot \nabla_b$$
  
   Where:
   - \( b' \) is the updated bias.
   - \( b \) is the current bias.
   - \( $$\eta$$ \) is the learning rate.
   - \( m \) is the number of examples in the mini-batch.
   - \( $$\nabla_b$$ \) is the gradient generated by a mini batch of the cost function with respect to the biases.

   The backpropagation algorithm begins by calculating the error, denoted as \( \delta \), in the final layer of the network. This is achieved by calling the static method 'delta' from the chosen cost function. The gradient for each weight and bias is then calculated recursively through the following equations:

   * The error in the final layer:
   
   $$\delta^L = \nabla_a C \odot \sigma'(z^L)$$
   
   where:
   - \( $$\delta^L$$ \) is the error in the final layer.
   - \( $$\nabla_a C$$ \) represents the gradient of the cost function \( C \) with respect to the activation \( a \).
   - \( $$\sigma'(z^L)$$ \) is the derivative of the activation function with respect to \( z^L \), the weighted input to the final layer.
   - \( $$\odot$$ \) denotes the Hadamard (element-wise) product.

   This is calculated in the cost classes, with the 'delta' method.

   * For each layer \( $$l$$ \) moving backward through the network, the error \( $$\delta$$ \) is updated based on the following layer's \( $$\delta$$ \):

   $$\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$$

   where:
   - \( $$\delta^l$$ \) is the error in layer \( l \).
   - \( $$(w^{l+1})^T$$ \) is the transpose of the weight matrix from layer \( l+1 \) to \( l \), which propagates the error backward.
   - \( $$\delta^{l+1}$$ \) is the error from the next layer \( l+1 \).
   - \( $$\sigma'(z^l)$$ \) is the derivative of the activation function with respect to \( $$z^l$$ \).

   Finally, the gradients for the biases and weights for each layer \( $$l$$ \) are calculated as:
   
   $$\frac{\partial C}{\partial b_j^l} = \delta_j^l$$

   $$\frac{\partial C}{\partial w_{jk}^l} = a_k^{l-1} \delta_j^l$$

   where:
   - \( $$\frac{\partial C}{\partial b_j^l}$$ \) is the partial derivative of the cost function with respect to the bias \($$ b_j^l$$ \) in layer \( $$l$$ \).
   - \( $$\frac{\partial C}{\partial w_{jk}^l}$$ \) is the partial derivative of the cost function with respect to the weight \( $$w_{jk}^l$$ \) in layer \( $$l$$ \).
   - \( $$a_k^{l-1}$$ \) is the activation from the previous layer \( $$l-1$$ \), used to calculate the gradient with respect to \( $$w_{jk}^l$$ \).
   - \( $$\delta_j^l$$ \) is the error term for neuron \( $$j$$ \) in layer \( $$l$$ \).
  
   The backprop() method returns these gradients, which are used to compute the overall gradient generated by update_mini_batch(), which in turned is used in the above update rules for weights and biases and changes the parameters of the network. 


7. **Variable Learning Schedule:** Variable learning schedule based on the "No improvement in 'n' epochs". If there is no improvement in the evaluation accuracy in `patience` (an optional parameter in the 'SGD' method) epochs and the 'variable_eta' flag is set to true, the learning rate \( \eta \) will be halved.
8. **Accuracy Calculation:** Accuracy is calculated with the 'accuracy' method in the Network class. This method returns how many predictions were correct in an epoch of training for training data and/or validation data. It is used when the monitoring flags in 'SGD' enable it. Furthermore, it can print the wrong predictions of the network when the network instance's internal boolean variable `showMistakes` is set to `True`. On top of that, it can display the propability guesses of all poissible labels of the wrong predictions using a softmax layer if the `softmax_layer` flag within the `accuracy()` method is set to `True`. 
9. **Cost Calculation:** Cost can also be calculated with the 'total_cost' method, which leverages the function definition of the cost function and returns the cost on training and/or validation data.
10. **Network Saving:** The network has a 'save' method that saves the sizes (list where each index is the number of neurons in the corresponding index layer), cost, weights, biases, and, if provided, test data into a JSON file. `SGD()` has a flag for saving the best network in all training time. When flagged with 'save_network', `SGD()` will keep track of the best network based on validation data accuracy and save the final best network and its corresponding test data (if provided) into a JSON file stored in the source code directory (`src`). In order to save the network with the test data, users must explicitly provide the network with test data by assigning it to the network instance's internal variable 'test_data'. This is done in the appropriate test functions. 
11. **Network Loading:** Outside of the Network class but still in the `my_net.py` script, there is a corresponding `load()` function that loads a network and its test data, if it exists.
12. **Optinal Softamx Final Layer:** I also added a softmax method so that the output of the Network can be interpreted as a probability distribution over the possible labels for a given input. This is method is called in `accuracy()` when the network instance's internal variable `showMistakes` is set to true and the functions `softmax_layer` flag is also `True`, and displayes the probability distribution of the network's erroneous classification. This is useful to see "how wrong" the network is when it makes a mistake. The method could also be called when evaluating an unseen and unlabeled sample data, but I have not coded such scenario.
13. **Feedforward:** The `Netwok` class contains a `feedfoward()` method that returns the activations in the output layer for a given input `x`. This is used in several  methods of the network class and would be the primary function for evaluating unseen and unlabeled real-world data. 

### siren_loader.py

This script is responsible for preprocessing audio data used in the Siren Classifier project. Below is a generic view of the functions and their purposes:

1. **`force_standard_size()`**:
   - Ensures that all audio files within the specified directory have the same duration.
   - Applies a smooth fade-out effect to the audio files to improve audio quality and consistency.

2. **`vectorized_result()`**:
   - Includes functions to convert string labels into numerical representations, facilitating easier processing by machine learning algorithms.

3. **`expand_data()`**:
   - Dynamically creates a new directory within the Siren Classifier project to store an expanded dataset, which can be beneficial for improving model performance through data augmentation.

4. **`extract_acoustic_features()`**:
   - Extracts relevant acoustic features from the audio files, such as spectral characteristics, and saves them as a list of tuples in the format (input, label) to a JSON file stored in a dynamically created directory. For MFCC ectraction, the function converts the data from `int16` to `float32` for compatibility with `librosa`. 

5. **`load_acoustic_features()`**:
   - Provides functionality to load acoustic features from the JSON file, making it easier to reuse the processed data in various parts of the project.

6. **`data_visualizer()`**:
   - Includes methods for visualizing the extracted acoustic features, helping to analyze the data and gain insights into its characteristics.

7. **`acoustic_data_wrapper()`**:
   - Implements a data wrapper that shuffles the dataset and returns lists of tuples for training, validation, and testing, ensuring that the model can learn effectively from diverse data samples. It has a `mode` optional variable that defines if the loaded data is from the vanilla dataset (`mode='standard' `) or from the expanded dataset(`mode='expand'`), with the aditional option for expandind the data just before loading with (`mode='expand'`). 


### plugins.py
To expand the audio dataset, I came up with simple plugin implementations that generate altered versions of each data point from the vanilla dataset. This is quite typical in Neural Network problems. I settled for a data expansion algorithm that does not leverage all of these plugins, although I would like to go back to this project and experiment with larger augmented datasets. I have the following plugins in this file: 

1. A Dynamic Compressor: Compressors reduce the dynamic range on an audio file according to a threshold and a ratio. They operate based on two primary parameters: threshold and ratio. The threshold is the specified level at which compression starts acting, meaning any sample that exceeds this level will be affected. The ratio determines the degree of compression applied to signals that exceed the threshold, expressed as a ratio such as 2:1 or 4:1. For example, with a ratio of 2:1, if a signal reaches 10 dB above the threshold, only 5 dB will be allowed to pass through, effectively reducing the excess signal by half. Each audio file in the vanilla dataset has a compressed version created with light settings in the expanded dataset.

2. A Shelving Filter: Shelving Filters are a type of filter whose frequency response is shaped like a sigmoid function, allowing it to boost or cut frequencies above or below a specified cutoff frequency, depending on the gain settings. In my data expansion algorithm, I generated one high cut, one high boost, one low cut, and one low boost for every file in the vanilla dataset. Although the difference is sublte, there is an expressive change in the file. 

3. A Stereo To Mono Converter: This plugin simply turns stereo files into mono files. It has two modes: 'sum' and 'split'. When the mode is 'sum', the mono file is determined by summing the left and right channels. When the mode is 'split', the plugin generates two mono files corresponding to the left and right channels. In my expansion algorithm, I am generating sums of the channels in the artificially expanded dataset. 

4. A Pitch Shifter: This plugin leverages librosa's implementation of pitch shifting. Pitch shifting is a process that can alter the frequency content of a file by a given number of semitones (defined using the Western A440 scale for musical notes). If the number of steps is positive, the plugin shifts the audio file that many steps up; else, if negative, it shifts the pitch lower by that number of semitones.

5. A Mask Generator: This plugin masks some information in the audio file. It supports temporal and frequency masking. A mask is a set of values that is set to 0 ("erased information"). If the settings are low enough, the file should barely be distinguishable from the original, but the actual digital representation should change. Lossy audio compressors take advantage of this phenomenon to erase non-critical information in audio files.

**Note:** Just like in `extract_acoustic_features()`, the files need to be in float format for `librosa` processing, so `plugins.py` also internally handles this conversion between `int16` and `float32`, potentially throwing erros if the input is in the wrong format. 

### tests.py

This file contains feature tests for some of the important functions in `siren_loader.py`. More importantly, it provides functions for training neural networks and testing trained networks against their test data. Below is a description of each test function:

1. `test_siren_loader(mode="standard")`
  - Tests loading data and visualizing samples using the specified mode (`standard`, `expanded`, or `expand`).
  - Displays the number of data points and visualizes a sample of the training data.

2. `test_mfcc()`
  - Reads an audio file, converts stereo to mono, normalizes the audio data, and extracts MFCC features.
  - Prints the shape of the extracted MFCCs.

3. `test_feature_list()`
  - Extracts acoustic features from standard data and provides debugging information such as unique labels, data tuple lengths, and feature vector length consistency.

4. `test_trained_NN(path, softmax=False)`
  - Loads a saved neural network and test data from the specified path and evaluates its accuracy on the test data.
  - Displays errors and the output actications/ probability distributions depending on the `softmax` flag.

5. `test_train_NN(mode="standard")`
  - Trains and tests the neural network using either `standard` or `expanded` acoustic data.
  - Dynamically determines the input size from the data and trains the network using specified hyperparameters.
  - Supports hyperparameter experimentation and saves the best model based on validation accuracy.


## Training and Final Results

Training a network requires a great deal of trial and error, so I experimented with hyperparameters in order to improve accuracy while monitoring several aspects of the network in each training epoch. I used the `save()` method to save the best networks in training. My best accuracies in training was with a network containing 2 hidden layers, the first with 500 neurons, and the second with 10 neurons (i.e. `sizes = [input.size(), 500, 10, output.size()]`), training for 150 epochs with a learning rate $$\eta$$ of 0.002, a patience of 8 for varying the learning rate, a mini-batch size of 40, and a regularization parameter $$\lambda$$ of 0.1, all while feeding the network expanded data. Training can take a few hours, and it was all done in my personal machine. 

## Future Improvements

This project and its documentation took me about 7 weeks, and although I would love to continue refining it, I have other responsibilities and projects that require my attention. However, I want to outline several enhancements I would like to pursue in the future:

### 1. Flexibility in Handling Various Audio Data and Labels

Currently, the classifier is designed for specific audio data and a fixed set of labels. I would like to improve the flexibility of the code so that it can handle any type of audio input and an arbitrary number of labels. This would make the classifier more versatile and applicable to a broader range of audio classification tasks.

### 2. Dataset Expansion Experiments

Expanding the dataset has proven to be quite relevant to the model's performance. Future work could involve experimenting with different methods of data augmentation to enrich the dataset, such as leveraging some existing plugins I coded but did not use in my expansion, like time-stretching, frequency and temporal masking, and pitch-shifting. This could help create more robust models that generalize better to unseen data.

### 3. Feature Extraction Process

Feature extraction is a critical step in any machine learning pipeline. I would like to experiment with different feature extraction techniques beyond the ones currently used. Exploring more advanced or domain-specific features could potentially lead to significant improvements in classification accuracy.

### 4. Handling and Raising User Errors

Ensuring the robustness of the code is vital for real-world applications. Further testing and refinement are needed to improve the error handling mechanisms. This includes anticipating potential user errors, such as incorrect input formats or missing data, and providing informative error messages or exceptions to guide users in resolving issues effectively.

### 5. Hyperparameter Tuning for Higher Accuracy

While the current model has achieved satisfactory results, there is always room for improvement. Further experimentation with hyperparameter tuning, such as learning rates, batch sizes, regularization parameters, and network architectures, could help push the model's performance to even higher levels of accuracy.

### 6. Experimenting With Different Training Oprtimizations

I would like to implement other training optimization techniques, such as Momentum-Based Stochastic Gradient Descent, Dropout, L1-Regularization, different rules for varying the learning schedule, etc. Although I have coded versions of a few of these for this project, their implementations were not robust enough to produce useful results. 

### 7. Deployment on a Microcontroller for Real-World Testing

One of the ultimate goals is to deploy a well-trained network onto a microcontroller. This would allow the classifier to be integrated into a vehicle for real-world audio monitoring tasks. By testing the classifier in this context, I can gather valuable field data and refine the model further to meet practical requirements and constraints, such as real-time processing and limited computational resources.



By addressing these areas in the future, I believe the classifier can be significantly improved and expanded to cover a wider range of applications. However, I am now focusing on studying more advanced Neural Network architectures, which, if applied in this context, will certainly outperform this dense layer-only-based implementation. 


