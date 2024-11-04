#### Libraries
# Standard library
import json
import sys
import numpy as np
import random
import datetime

"""
This implementation defines a neural network classifier for audio data based on Michael Nielsen's 
code from "Neural Networks and Deep Learning," available at 
https://github.com/mnielsen/neural-networks-and-deep-learning (this code is based on net2.py). I have
adapted it to work with my data processing pipeline and included several proposed features from Nielsen's 
book, as well as additional custom features to enhance performance. This implementation is far from optimal
as this entire project serves as a deep dive into neural networks and a personal challenge for me. So far, my
 best accuracy on the expanded data is 96.67% on unseen training data, indicating a strong model performance.

Note: This implementation does not rely on libraries like tensorflow or pytorch. 
"""

#### Define cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output `a` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)
    
#Define the Network
class Network(object):

    def __init__(self, sizes, cost_func=CrossEntropyCost):
        self.sizes = sizes #Set the size of the network
        self.num_layers = len(sizes) #Get the number of layers from the sizes
        self.default_weight_initializer() #Initialize the weights
        self.cost=cost_func #Define cost function
        self.test_data = None #No test data unless it is provided for saving the network
        self.showMistakes = False #Flag for printing mistakes in the accuracy function

    
    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        
    def feedforward(self, a):
        """Return the output of the network when `a` is the input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            patience=10, 
            variable_eta=False,
            save_network=False):
        """
            Implements Stochastic Gradient Descent (SGD) for training a neural network.

            This is the main function for training the network. It handles shuffling the training data,
            creating mini-batches, updating the weights and biases of the network, and optional monitoring 
            of the training and evaluation metrics, including cost and accuracy.

            Parameters:
            - training_data: list of tuples (input, output) representing the training dataset, where
            input and output should be numpy arrays with shapes corresponding to the network's input 
            and output layers.
            - epochs: int, the number of times to iterate over the training data.
            - mini_batch_size: int, the number of training examples to use in each mini-batch.
            - eta: float, the learning rate for weight updates.
            - lmbda: float, optional, the regularization parameter (default is 0.0).
            - evaluation_data: list of tuples (input, output), optional, the data used for evaluating 
            the network during training.
            - monitor_evaluation_cost: bool, optional, whether to monitor the evaluation cost (default is False).
            - monitor_evaluation_accuracy: bool, optional, whether to monitor the evaluation accuracy (default is False).
            - monitor_training_cost: bool, optional, whether to monitor the training cost (default is False).
            - monitor_training_accuracy: bool, optional, whether to monitor the training accuracy (default is False).
            - patience: int, optional, the number of epochs with no improvement in evaluation accuracy 
            before reducing the learning rate (default is 10).
            - variable_eta: bool, optional, whether to adjust the learning rate based on performance (default is False).
            - save_network: bool, optional, whether to save the best network's parameters after training (default is False).

            Returns:
            - evaluation_cost: list of floats, the recorded evaluation costs over epochs.
            - evaluation_accuracy: list of floats, the recorded evaluation accuracies over epochs.
            - training_cost: list of floats, the recorded training costs over epochs.
            - training_accuracy: list of floats, the recorded training accuracies over epochs.

            Modifies:
            - Updates the network's weights and biases based on the training data.
            - Optionally saves the best performing network to a file.

            Raises:
            - AssertionError: If any input or output data in the training or evaluation datasets do not match
            the expected shapes of the network's input and output layers.
            """
        #Ensure all data is valid and check for evaluation data
        if evaluation_data:
            assert all(isinstance(e, tuple) and len(e) == 2 for e in evaluation_data), "Invalid evaluation data"
            n_data = len(evaluation_data)
        n = len(training_data)

        for x, y in training_data:
            try:
                assert x.shape == (self.sizes[0], 1), f"Incorrect input shape of {x.shape} instead of ({self.sizes[0]}, 1)"
                assert y.shape == (self.sizes[-1], 1), f"Incorrect output shape of {y.shape} instead of ({self.sizes[-1]}, 1)"
            except AssertionError as e:
                print(f"Data shape mismatch: {e}")
                raise

        print(f"Training data length: {len(training_data)}")

        # Initialize arrays to store evaluation and training metrics
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        # Initialize variables for early stopping based on no improvements
        improvement_counter = 0
        best_eval_ac = 0  # Track the highest accuracy, not lowest cost
        best_network_data = None  # Dictionary to store the best network's data
        eta_counter = 0

        for j in range(epochs):
            # Shuffle data
            random.shuffle(training_data)

            # Create list of mini_batches
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            #Update mini batches
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))

            print("\nEpoch %s training complete" % j)

            # Get the training cost for this epoch
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")

            # Get the accuracy on the training data
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=False)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {n} ({(100 * accuracy / n):.3f}%)")

            # Get the evaluation cost for this epoch
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")

            # Get the accuracy on the evaluation data
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=False)
                evaluation_accuracy.append(accuracy)
                eval_percentage = 100 * accuracy / n_data
                print(f"Accuracy on evaluation data: {accuracy} / {n_data} ({eval_percentage:.3f}%)\n")

                # Check if there is improvement in evaluation accuracy
                if accuracy > best_eval_ac:
                    best_eval_ac = accuracy
                    improvement_counter = 0
                    print(f"New best eval accuracy: {eval_percentage:.3f}%")

                    # Update best_network_data dictionary with the current best network
                    best_network_data = {
                        "sizes": self.sizes,
                        "weights": [w.tolist() for w in self.weights],
                        "biases": [b.tolist() for b in self.biases],
                        "cost": str(self.cost.__name__)
                    }
                else:
                    improvement_counter += 1
                    print(f"No improvement in {improvement_counter} epoch(s)")

                    # Halve eta if no improvement for 'patience' epochs
                    if improvement_counter >= patience:
                        if variable_eta:
                            eta /= 2
                            eta_counter += 1
                            print(f"Halving eta to {eta}")
                            improvement_counter = 0  # Reset the counter after halving eta (If stopping early)

        # Save the best network after training completes, if flagged
        if save_network and best_network_data:
            timestamp = datetime.datetime.now().strftime("%d_%m_%H_%M")
            filename = f"{timestamp}_best_network_ac_of_{(100 * best_eval_ac / n_data):.2f}.json"
            self.save(filename, self.test_data)
            print(f"Best network saved as {filename} with accuracy: {(100 * best_eval_ac / n_data):.2f}%")

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy


    def update_mini_batch(self, mini_batch, eta, lmbda, n):

        """
        Updates the network's weights and biases using the gradients calculated 
        from the mini-batch of training data.

        This function performs a single step of stochastic gradient descent by 
        calculating the gradients of the cost function with respect to the 
        network's parameters (weights and biases) for each training example 
        in the mini-batch, and then applying the update rule to adjust the 
        parameters.

        Parameters:
        - mini_batch: list of tuples (input, output) representing a mini-batch 
        of training data, where input and output should be numpy arrays with 
        shapes corresponding to the network's input and output layers.
        - eta: float, the learning rate for updating weights and biases.
        - lmbda: float, the regularization parameter used in the weight update 
        rule to prevent overfitting.
        - n: int, the total number of training examples, used for calculating 
        the weight update with regularization.

        Returns:
        - None

        Modifies:
        - Updates the network's weights and biases based on the gradients computed 
        from the mini-batch.
        """

        #Initialize the gradiens for the weights and biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #Loop through mini_batch
        for x, y in mini_batch:

            #Call backprop and get the gradients for this x, y input
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            #Update nablas
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #Use the nablas in the update rule for weights and biases
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (eta / len(mini_batch)) * nb
                    for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        """
        Performs backpropagation to calculate the gradients of the cost 
        function with respect to the network's weights and biases.

        This function computes the gradients for a single training example 
        by first performing a forward pass to compute the activations for 
        each layer, then executing a backward pass to calculate the error 
        and gradients for the weights and biases.

        Parameters:
        - x: numpy array, the input data for the training example with shape 
        corresponding to the network's input layer.
        - y: numpy array, the expected output (target) for the training 
        example with shape corresponding to the network's output layer.

        Returns:
        - tuple: A tuple containing two lists:
            - nabla_b: list of numpy arrays, the gradients of the biases 
            for each layer.
            - nabla_w: list of numpy arrays, the gradients of the weights 
            for each layer.

        Modifies:
        - None: This function does not modify any attributes of the class.

        """

        #Initialize the gradients for this training example's gradients
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
            
        #Forward pass while keeping track of all activations and z's
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b 
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        #Backwards pass
        delta = self.cost.delta(zs[-1], activations[-1], y) #Get the error from the last layer
        nabla_b[-1] = delta #Build the last bias nabla according to derived equations
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #Build the last weigth nabla according to equations

        #Loop through the rest of the layers and generate all nablas according to the golden rule of backprop
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)
    
    def softmax(self, output_activations):
        """
        Apply the softmax function to the output activations.

        Parameters:
        -output_activations: list, consists of the outout layer's activations

        Returns:
        - softmax_values: float, a list of the probabilities (in percentage) of the predicted labels

        Modifies:
        - Nothing
        """
        # Subtract the max value for numerical stability
        exp_activations = np.exp(output_activations - np.max(output_activations))
        # Compute softmax values
        softmax_values = exp_activations / np.sum(exp_activations)
        return softmax_values

    def accuracy(self, data, convert=False, softmax_layer=False):
        """
        Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        Parameters:
        -output_activations: list of tuples, either training data or evaluation data

        Returns:
        - correct_cout: int, number of correct predictions on the data labels

        Modifies:
        - prints information to the terminal if showMistakes is true 
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            # Convert y from one-hot encoding to scalar if needed
            results = [(np.argmax(self.feedforward(x)), np.argmax(y) if isinstance(y, np.ndarray) else y)
                    for (x, y) in data]
            
        # Track the number of correct predictions
        correct_count = 0

        # Iterate over data and results to check for mistakes and log neuron values
        for (x, y), (predicted, actual) in zip(data, results):
            # Get the output activations from the final layer
            output_activations = self.feedforward(x)
            
            if predicted != actual:
                # Calculate predicted value and actual value
                predicted_value = output_activations[predicted]
                actual_value = output_activations[actual] if isinstance(y, np.ndarray) else None
                
                if self.showMistakes:
                    if softmax_layer:
                        probabilities = self.softmax(output_activations)
                        print(f"\nIncorrect Prediction: Predicted label: {predicted}, "
                            f"Actual label: {actual} \n")
                        print(f"Softmax Probabilities: \n{probabilities*100}")
                    else:
                        print(f"Incorrect Prediction: Predicted label: {predicted} with activation {predicted_value}, "
                            f"Actual label: {actual} with activation {actual_value}")
            else:
                correct_count += 1
            
        return correct_count

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.

        Parameters:
        - data: list of tuples, either training or evaluation data

        Returns:
        - cost: float, the cost associated with the dataset with respect to the network's parameters. 
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """
        Save the neural network and optionally the test data to the file ``filename``.

        Parameters:
        - filename: string, name of the JSON file that will contain saved data from the network

        Modifies:
        - src directory: Adds a JSON file containing information on the saved network
        """
        network_data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__)
        }
        
        if self.test_data is None:
            print("Warning: No test data available. Saving only network data.")
            data = network_data
        else:
            # Prepare the test data in a serializable format
            serializable_test_data = [(x.tolist(), y.tolist()) for x, y in self.test_data]
            
            # Save as a tuple (network_data, test_data)
            data = (network_data, serializable_test_data)
        
        with open(filename, "w") as f:
            json.dump(data, f)

        print(f"Saved network to {filename}!")

#### Loading a Network

import json
import numpy as np
import sys

def load(filename):
    """
    Load a neural network and optionally test data from the file ``filename``. 
    Returns a tuple (Network, test_data). If test data is not present, test_data will be None.
    
    Parameters:
    - filename: string, path to the file containing the network data

    Returns: 
    - A tuple (network instance, test data) if test data is found
    - OR a network instance if test data is None
    """

    # Open and load the JSON data
    with open(filename, "r") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        # Data contains only the network data
        network_data = data
        test_data = None
        print("Warning: No test data present in JSON file. Loading data as a network dictionary and returning an instance of 'Network'.")
            # Reconstruct the network
        cost = getattr(sys.modules[__name__], network_data["cost"])
        net = Network(network_data["sizes"], cost_func=cost)
        net.weights = [np.array(w) for w in network_data["weights"]]
        net.biases = [np.array(b) for b in network_data["biases"]]
        return net
    else:
        # Data contains both network data and test data
        network_data, test_data_raw = data
        
        # Reconstruct the test data as tuples of numpy arrays
        test_data = [(np.array(x), np.array(y)) for x, y in test_data_raw]

    # Reconstruct the network
    cost = getattr(sys.modules[__name__], network_data["cost"])
    net = Network(network_data["sizes"], cost_func=cost)
    net.weights = [np.array(w) for w in network_data["weights"]]
    net.biases = [np.array(b) for b in network_data["biases"]]

    return net, test_data

#Sigmoid functions

def sigmoid(z):
    z = np.clip(z, -500, 500)  # Limit the values to avoid overflow
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    z = np.clip(z, -500, 500)  # Limit the values to avoid overflow
    return sigmoid(z)*(1-sigmoid(z))



            