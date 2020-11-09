# %load mnist_loader.py
"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip
import random

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename = 'mnist.pkl.gz'):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper(filename = 'mnist.pkl.gz', output_size = (28, 28)):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data(filename)
    training_inputs = [np.reshape(x, output_size) for x in tr_d[0]]
    #training_inputs = [np.reshape(x, (28, 28)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, output_size) for x in va_d[0]]
    #validation_inputs = [np.reshape(x, (28, 28)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, output_size) for x in te_d[0]]
    #test_inputs = [np.reshape(x, (28, 28)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j, num_classes = 10):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((num_classes, 1))
    e[j] = 1.0
    return e

def unvectorize_result(vec):
    '''
    Gets the scalar version of a label from a vectorized label
    Note: I wrote this function myself
    '''
    vec_comp = np.ravel(vec)
    return vec_comp.tolist().index(1.0)

if __name__ == "__main__":
    '''
    This statement generates the subplots for Task 1, part 3
    '''
    
    train, val, test = load_data_wrapper("mnist.pkl.gz")
    train_list = list(train)
    val_list = list(val)
    test_list = list(test)

    total_list = train_list + val_list + test_list

    # Build the labels for every portion of the data
    total_labels_train = [ unvectorize_result(train_list[i][1]) for i in range(0, len(train_list)) ]
    total_labels_test = [test_list[i][1] for i in range(0, len(test_list))]
    total_labels_val = [val_list[i][1] for i in range(0, len(val_list))]
    # Draw samples from all of the datasets
    total_labels = total_labels_train + total_labels_val + total_labels_test

    total_samples = [total_list[i][0] for i in range(0, len(total_list))]

    len_vector = len(total_labels)

    label_indices = {}
    for i in range(0, 10): # Create empty lists each keyed on the label needed
        label_indices[i] = []

    # Build individual vectors with the indices for each label inside of them
    for i in range(0, len(total_labels)):
        label_indices[ total_labels[i] ].append(i)

    # Pick random elements
    random_pics = [[] for i in range(0, 10)]
    for i in range(0, 10):
        r_nums = random.sample(label_indices[i], 5) # Chooses 5 random samples

        for rn in r_nums: # Goes and gets random samples, puts them in one spot
            random_pics[i].append(total_samples[rn])

    fig, axs = plt.subplots(10, 5, sharey = True, sharex = True)
    fig.suptitle("MNIST Data Samples")

    for i in range(0, 10): # Over the rows
        for j in range(0, 5): # Over the cols
            axs[i][j].imshow(random_pics[i][j], cmap = "gray")

    plt.show()