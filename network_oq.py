# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import time

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

from mnist_loader import load_data_wrapper, unvectorize_result

class Network(object):

    def __init__(self, sizes, activation = "sigmoid", alpha = 0):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        if not (activation == "sigmoid" or activation == "RELU" or activation == "leakyRELU"):
            print("Please use one of the available activation functions: sigmoid, RELU, or leakyRELU")
        self.act_fn = activation
        self.alpha = alpha # Parameter for leakyRELU (only used if activation is leakyRELU)

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            # Chooses from different activation functions
            if (self.act_fn == "sigmoid"):
                a = sigmoid(np.dot(w, a)+b)
            elif (self.act_fn == "RELU"):
                a = RELU(np.dot(w, a) + b)
            elif (self.act_fn == "leakyRELU"):
                a = leakyRELU(np.dot(w, a) + b, self.alpha)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, plot = False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)
        accs = []

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                # Add to accuracies list: 
                accs.append(self.evaluate(test_data) / n_test)
    
            else:
                accs.append(self.evaluate(test_data) / n_test)
                print("Epoch {} complete".format(j))

        if (plot): # Plots the Convergence Curve if specified
            print("Accuracy at Epoch {} = {:.5f}".format(epochs, accs[-1]))

            plt.plot(range(1, len(accs) + 1), accs, linestyle = '-', color = 'green')
    
            if (self.num_layers > 3):
                plt.title("Convergence Curve: {} Total Layers".format(self.num_layers))
            else:
                plt.title("Convergence Curve (mbatch={}), (eta={})".format(mini_batch_size, eta))

            plt.xlabel("Epoch")
            plt.ylabel("Accuracy after Epoch")
            plt.show()

    def adaptiveSGD(self, training_data, mini_batch_size, eta, validation_data, max_epochs = 100,
            threshold = 0.01, acc_thresh = 0, test_data=None, plot = False):
        '''
        Runs SGD and stops it when the average change over the last 5 epochs is less than 
            the threshold

        Arguments:
        ----------
        training_data: 
        mini_batch_size: int
            - Size of batch to consider for each run of SGD
        eta: float
            - Learning rate
        validation_data: 
            - Data to check with each iteration
        max_epochs: integer, optional
            - Maximum number of epochs before breaking from the algorithm
            - Allows you to limit how long you'll let the algorithm run before stopping
        threshold: float, optional
        acc_thresh: float, optional
            - Default: 0
            - Must be in the range [0, 1]
            - If the model is not above this accuracy but the difference is less than
                the threshold variable, then we continue the epochs
        test_data: 
        plot: bool, optional
        '''

        # Transform training and validation data to lists:
        training_data = list(training_data)
        n = len(training_data)

        validation_data = list(validation_data)
        n_val = len(validation_data)
        val_accs = []

        epochs = 0 # Counts number of epochs

        if test_data: # Check if test data was provided
            test_data = list(test_data)
            n_test = len(test_data)

        while True:
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            # Evaluate on validation data, push to validation accuracy:
            val_accs.append(self.evaluate(validation_data) / n_val)
            epochs += 1

            # Check if mean difference was less than threshold
            if (epochs >= 5):
                diff_sum = sum([(val_accs[i] - val_accs[i - 1]) for i in np.arange(-1, -5, -1)])
                avg_acc = sum([val_accs[i] for i in np.arange(-1, -6, -1)]) / 5
                if (diff_sum <= threshold and avg_acc >= acc_thresh): # Check if diff sum below threshold - if so, break
                    break

            if (epochs >= max_epochs):
                break

        if test_data:
            # Add to accuracies list: 
            print("Testing Accuracy at Epoch {}    = {:.5f}".format(epochs, (self.evaluate(test_data) / n_test)))

        if (plot): # Plots the convergence Curve if Specified
            print("Validation Accuracy at Epoch {} = {:.5f}".format(epochs, val_accs[-1]))

            plt.plot(range(1, len(val_accs) + 1), val_accs, linestyle = '-', color = 'purple')
            plt.title("Validation Convergence Curve (mbatch={}), (eta={})".format(mini_batch_size, eta))
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy after Epoch")
            plt.show()

        return epochs

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        count = 0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            if (self.act_fn == "sigmoid"):
                activation = sigmoid(z)
            elif (self.act_fn == "RELU"):
                activation = RELU(z)
            elif (self.act_fn == "leakyRELU"):
                activation = leakyRELU(z, self.alpha)

            activations.append(activation)
        
        if (self.act_fn == "sigmoid"):
            delta_mult = sigmoid_prime(zs[-1])
        elif (self.act_fn == "RELU"):
            delta_mult = RELU_prime(zs[-1])
        elif (self.act_fn == "leakyRELU"):
            delta_mult = leakyRELU_prime(zs[-1], self.alpha)
        
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            delta_mult
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            if (self.act_fn == "sigmoid"):
                sp = sigmoid_prime(z)
            elif (self.act_fn == "RELU"):
                sp = RELU_prime(z)
            elif (self.act_fn == "leakyRELU"):
                sp = leakyRELU_prime(z, self.alpha)

            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        t_list = [(x, y) for (x, y) in test_data]
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    z_prec = np.array(z, dtype=np.float128) # Make float more precise to avoid overflow
    return 1.0/(1.0+np.exp(-z_prec))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    z_prec = np.array(z, dtype=np.float128) # Make float more precise to avoid overflow
    return sigmoid(z_prec)*(1-sigmoid(z_prec))

# Extra RELU functions
def RELU(z):
    ''' RELU Activation function for Bonus'''
    activations = (np.array([max(0, z[i][0]) for i in range(0, z.shape[0])]))
    return activations.reshape(z.shape)

def RELU_prime(z):
    ''' RELU derivative '''
    z_flat = z.ravel()
    primes = []
    for z_e in z_flat:
        primes.append(1 if (z_e > 0) else 0)
    return (np.array(primes)).reshape(z.shape)

def leakyRELU(z, alpha):
    ''' leakyRELU Activation function for Bonus'''
    activations = (np.array([max(alpha * z[i][0], z[i][0]) for i in range(0, z.shape[0])]))
    return activations.reshape(z.shape)

def leakyRELU_prime(z, alpha):
    ''' leakyRELU derivative '''
    z_flat = z.ravel()
    primes = []
    for z_e in z_flat:
        primes.append(1 if (z_e > 0) else alpha)
    return (np.array(primes)).reshape(z.shape)