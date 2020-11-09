import time
import numpy as np
from network_oq import Network
from mnist_loader import load_data_wrapper, vectorized_result

# This script contains functions to load the XOR data and train the model on XOR

def make_XOR_train():
    '''
    Generates the XOR training data
    No arguments

    Returns:
    --------
    XOR_train: zip
        - Returns data in same exact format as in mnist_loader
        - Labels are hot labels (as with the training data in mnist_loader)
    '''
    
    XOR_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    XOR_labels = [0, 1, 1, 0]

    XOR_points = [np.reshape(i, (2, 1)) for i in XOR_points]
    XOR_labels = [vectorized_result(i, 2) for i in XOR_labels]

    return zip(XOR_points, XOR_labels)

def make_XOR_test():
    '''
    Generates the XOR testing data
    No arguments

    Returns:
    --------
    XOR_test: zip
        - Returns data in same exact format as in mnist_loader
        - Labels are scalars (0 or 1)
    '''
    
    XOR_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    XOR_labels = [0, 1, 1, 0]

    XOR_points = [np.reshape(i, (2, 1)) for i in XOR_points]

    return zip(XOR_points, XOR_labels)

# This function will help us train the network on XOR
def train_xor(mbs = 10, eta = 3.0, layers = [2, 4, 2], max_epochs = 1000):
    xor_train = make_XOR_train()
    xor_val   = make_XOR_test() # Use identical validation and test data
    xor_test  = make_XOR_test()
    
    base_net = Network(layers, activation = "sigmoid")
    
    # Runs 
    st = time.time()
    eps = base_net.adaptiveSGD(training_data = xor_train, mini_batch_size = mbs,
            validation_data = xor_val, max_epochs = max_epochs, eta = eta, 
            test_data = xor_test, threshold = 0, acc_thresh = 1, plot = True)
    print("Time training:", time.time() - st)

if __name__ == "__main__":
    # Change these arguments to run the Task 4 trials from this script
    train_xor(mbs = 4, eta = 10, layers = [2, 2, 2])
