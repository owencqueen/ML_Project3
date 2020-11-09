# This code has been adopted from Dr. Qi's examples from Canvas
import numpy as np
import sys
import time
import util # local module


def knn(Tr, yTr, Te, k):
    '''
    Runs the kNN classification algorithm on given data
    Note: Taken from Dr. Qi's code posted on Canvas

    Arguments:
    ----------
    Tr: np array
        - Training data samples (excluding labels)
    yTr: np array
        - Training data labels
    Te: np array
        - Testing data
    k: int
        - k hyperparameter in kNN

    Returns:
    --------
    y: np array
        - List of the predicted labels from the model

    '''
    # training process - derive the model

    classes = np.unique(yTr)   # get unique labels as dictionary items
    classn = len(classes)      # number of classes
    ntr, _ = Tr.shape
    nte, _ = Te.shape
    
    y = np.zeros(nte)
    knn_count = np.zeros(classn)
    for i in range(nte):
        test = np.tile(Te[i,:], (ntr, 1))       # resembles MATLAB's repmat function
        dist = np.sum((test - Tr) ** 2, axis = 1) # calculate distance
        idist = np.argsort(dist)    # sort the array in the ascending order and return the index
        knn_label = yTr[idist[0:k]]
        for c in range(classn):
            knn_count[c] = np.sum(knn_label == c)
        y[i] = np.argmax(knn_count)
        
    return y 

def mpp(Tr, yTr, Te, cases, P):
    '''
    Runs maximum posterior probability classifier on given data
    Note: Taken from Dr. Qi's code posted on Canvas

    Arguments:
    ----------
    Tr: np array
        - Training data samples (excluding labels)
    yTr: np array
        - Training data labels
    Te: np array
        - Testing data
    cases: int
        - Options: 1, 2, or 3
        - Corresponds to different cases that we discussed in class
    
    Returns:
    --------
    y: np array
        - List of the predicted labels from the model
    '''
    # training process - derive the model
    covs, means = {}, {}     # dictionaries
    covsum = None

    classes = np.unique(yTr)   # get unique labels as dictionary items
    classn = len(classes)    # number of classes
    
    for c in classes:
        # filter out samples for the c^th class
        arr = Tr[yTr == c]  
        # calculate statistics
        covs[c] = np.cov(np.transpose(arr))
        means[c] = np.mean(arr, axis=0)  # mean along the columns
        # accumulate the covariance matrices for Case 1 and Case 2
        if covsum is None:
            covsum = covs[c]
        else:
            covsum += covs[c]
    
    # used by case 2
    covavg = covsum / classn
    # used by case 1
    varavg = np.sum(np.diagonal(covavg)) / classn
            
    # testing process - apply the learned model on test set 
    disc = np.zeros(classn)
    nr, _ = Te.shape
    y = np.zeros(nr)            # to hold labels assigned from the learned model

    for i in range(nr):
        for c in classes:
            if cases == 1:
                edist2 = util.euc2(means[c], Te[i])
                disc[c] = -edist2 / (2 * varavg) + np.log(P[c] + 0.000001)
            elif cases == 2: 
                mdist2 = util.mah2(means[c], Te[i], covavg)
                disc[c] = -mdist2 / 2 + np.log(P[c] + 0.000001)
            elif cases == 3:
                mdist2 = util.mah2(means[c], Te[i], covs[c])
                disc[c] = -mdist2 / 2 - np.log(np.linalg.det(covs[c])) / 2 + np.log(P[c] + 0.000001)
            else:
                print("Can only handle case numbers 1, 2, 3.")
                sys.exit(1)
        y[i] = disc.argmax()
            
    return y    
    
