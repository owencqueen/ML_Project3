import time
import numpy as np
from mnist_loader import load_data_wrapper, unvectorize_result
from mylearn import knn

# Load in the data:
train, val, test = load_data_wrapper("mnist.pkl.gz")

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

# Run the kNN multiple trials:
k_vals = [5, 10, 20, 35, 50, 75, 100, 150]
for k in k_vals:
    start_time = time.time()
    predictions = knn(Tr = Xtrain, yTr = Ytrain_scalar, Te = Xtest, k = k)
    print("KNN Run time:", time.time() - start_time, "\n")

    np.save("kNN_test/pred_knn={}.npy".format(k), predictions)