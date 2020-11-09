import time
import numpy as np
import sys, os

sys.path.append(os.getcwd() + "/..") # Add parent directory to include path

from mnist_loader import load_data_wrapper, unvectorize_result
from dim_reduce import pca

# Runs the PCA development for Bonus in Task 2

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
    # Run the PCA:
    tolerances = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
    for tol in tolerances:
        start_time = time.time()
        Xtrain_pca, Xtest_pca, error_rate = pca(Xtrain, Xtest, tolerance = tol, return_both = True)
        print("PCA time (tol = {}):".format(tol), time.time() - start_time)
        print("Error: ", error_rate)
        print("Eigenvectors kept = {}".format(Xtrain_pca.shape[1]))
        print("")

        # Saves the files to pca folder
        np.save("pca_mnist/train/train_tol={:.3f}.npy".format(tol), Xtrain_pca)
        np.save("pca_mnist/test/test_tol={:.3f}.npy".format(tol), Xtest_pca)

if __name__ == "__main__":
    main()
