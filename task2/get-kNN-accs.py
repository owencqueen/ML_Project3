import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.getcwd() + "/..") # Add parent directory to include path

from mnist_loader import load_data_wrapper, unvectorize_result
from util import accuracy_score

# Load in the data:
train, val, test = load_data_wrapper("../mnist.pkl.gz")

train = list(train)
Xtrain        = np.array([np.ravel(sample[0]) for sample in train])
Ytrain_vector = np.array([np.array(sample[1]) for sample in train])
Ytrain_scalar = np.array([unvectorize_result(sample[1]) for sample in train])

val = list(val)
Xval        = np.array([np.ravel(sample[0]) for sample in val])
Yval_scalar = [sample[1] for sample in val]

test = list(test)
Xtest        = np.array([np.ravel(sample[0]) for sample in test])
Ytest_scalar = [sample[1] for sample in test]

def gen_knn_acc_plot(k_vals):
    '''
    Arguments:
    ----------
    k_vals: list of ints
        - Values of k to generate plots for
        - Must be saved in the kNN_mnist directory as "pred_knn={}.npy" 
            where {} is the k value

    Returns:
    --------
    No explicit return value
    '''

    overall_vec = []

    for k in k_vals: # Iterate over k values
        k_pred = np.load("kNN_mnist/pred_knn={}.npy".format(k))
        knn_classwise_a, knn_overall_a = accuracy_score(y = Ytest_scalar, y_model = k_pred)
        print("kNN (k = {}) Overall Accuracy:".format(k), knn_overall_a)

        overall_vec.append(knn_overall_a)
    
    plt.plot(k_vals, overall_vec, color = "green")
    plt.xlabel("k value")
    plt.ylabel("Accuracy")
    plt.title("kNN k Value vs. Accuracy")

    plt.show()

if __name__ == "__main__":
    kv = [5, 10, 20, 35, 50, 75, 100, 150]
    gen_knn_acc_plot(kv)

