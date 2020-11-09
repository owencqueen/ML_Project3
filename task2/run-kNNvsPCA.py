import time
import numpy as np
import sys, os

sys.path.append(os.getcwd() + "/..") # Add parent directory to include path

from mnist_loader import load_data_wrapper, unvectorize_result
from mylearn import knn
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

def main():
    # Run the kNN multiple trials with differing PCA tolerance:
    k = 20
    tolerances = [0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025, 0.01]

    for tol in tolerances:

        pca_trainX = np.load("pca_mnist/train/train_tol={:.3f}.npy".format(tol))
        pca_testX = np.load("pca_mnist/test/test_tol={:.3f}.npy".format(tol))

        print("Tolerance level:", tol)

        start_time = time.time()
        predictions = knn(Tr = pca_trainX, yTr = Ytrain_scalar, Te = pca_testX, k = k)
        print("KNN Run time:", time.time() - start_time, "\n")

        knn_classwise_a, knn_overall_a = accuracy_score(y = Ytest_scalar, y_model = predictions)
        print("kNN (k = {}) Overall Accuracy:".format(k), knn_overall_a)
        print("")

        np.save("pca_mnist/kNN_pca/pred_knn={}_tol={:.3f}.npy".format(k, tol), predictions)

if __name__ == "__main__":
    main()