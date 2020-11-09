import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def split_data_by_class(train_data, labels):
    '''
    Splits the training data into its individual classes

    Arguments:
    ----------
    train_data: pandas dataframe
        - Dataframe that you want to split
        - Assumption is that labels are contained in the last row (row -1)
    
    Returns:
    --------
    split_data: dictionary
        - Keyed on labels found in last row
        - Groups the data by label found in last row
        - Note: split data does still contain labels
    '''

    labels_unique = set(labels) # Convert to a set - will isolate unique values
    labels_unique = list(labels_unique) 

    split_data = {l:[] for l in labels_unique}  

    for l in labels_unique:
        split_data[l] = train_data.loc[lambda d: d.iloc[:,-1] == l, :]

    return split_data

def project_data(w, output_data):
    '''
    Projects the output_data onto the subspace give by w

    Arguments:
    ----------
    w: np.array
        - Subspace on which to project the output data
    output_data: pandas DataFrame
        - Data which to project

    Returns:
    --------
    pd_projected: pandas dataframe
        - Projected data
    '''

    #output_labels = output_data.iloc[:,-1]
    #output_data_np = output_data.iloc[:, 0:-1].to_numpy()
    #projected_np_data = np.transpose(np.matmul(w, np.transpose(output_data_np)))
    
    #pd_projected = pd.DataFrame(data = projected_np_data)

    #pd_projected["label"] = output_labels

    return np.transpose(np.matmul(w, np.transpose(output_data)))

def pca(Xtrain, output_data, num_dim = -1, tolerance = 0.15, return_both = False):
    '''
    Arguments:
    ----------
    data: pandas DataFrame
        - Will be data that we reduce
        - Labels must be left in, will be returned in final dataframe
    output_data: pandas DataFrame
        - Data that will be projected with computed principal components
        - Equivalent to testing data
    num_dim: positive int
        - Default: -1
        - If not left to default, this specifies the number of dimensions to keep in pca
        - Must be <= number of dimensions in the data
    tolerance: float [0, 1]
        - Specifies the maximum error that we will allow in our reduction
    return_both: bool, optional
        - Default: False
        - If true, returns a tuple of projected data and projected output_data (in that order)

    Returns:
    --------
    trimmed: pandas DataFrame
        - Output data projected onto reduced dimensions
    Return value may be tuple of projected data and projected output_data if return_both = True
    '''

    # Calculate covariance matrix
    #data_as_np = data.iloc[:, 0:-1].to_numpy() # Convert to numpy array

    #cov_mat = np.cov(np.transpose(data_as_np)) # Get covariance matrix
    cov_mat = np.cov(np.transpose(Xtrain)) # Get covariance matrix

    # Calculate eigenvalues for the cov matrix
    evals, evecs = np.linalg.eig(cov_mat)

    # Note: evecs are stored in columns corresponding to each eval
    #   - Access eigenvector corresponding to evals[i] at evecs[:,i]

    # Sort eigenvalues from largest to smallest
    order_eigen = np.flip(np.argsort(evals))
    ordered_evals = [evals[i] for i in order_eigen]

    sum_evals = sum(ordered_evals)

    # Choose eigenvalues to keep
    if (num_dim > -1): # Prioritize based on num_dim
        to_keep = order_eigen[0:num_dim]
        
        # Calculate eigenvectors for each eigenvalue we keep
        keep_evecs = [evecs[:,i] for i in to_keep]

        # Get error rate - eigenvalues not kept as specified
        error_rate = np.sum( [ordered_evals[num_dim:]] ) / sum_evals

    else: # Else, we need to keep based on tolerance
        error_rate = 0
        #num_keep = 0
        num_keep = len(order_eigen)

        # Iterate backwards on ordered eigenvalues
        for i in np.arange(len(order_eigen) - 1, -1, -1):
            error_rate += ordered_evals[i] / sum_evals
            
            # If our error rate is above tolerance, adjust it and break
            if (error_rate >= tolerance):
                error_rate -= ordered_evals[i] / sum_evals
                break

            #num_keep += 1 # Need to increment number of eigenvectors we want to keep
            num_keep -= 1

        keep_evecs = [evecs[:,i] for i in order_eigen[0:num_keep]]

    # Ensure that eigenvectors are normalized
    #   Note: The numpy algorithm automatically normalizes the eigenvectors

    # Project data onto basis vectors

    w = np.array(keep_evecs)

    if (return_both):
        return (project_data(w, Xtrain), project_data(w, output_data), error_rate)
    else:
        return (project_data(w, output_data), error_rate)

# For this project, do not try to run this function:
def fld(data, output_data, return_both = False):
    '''
    Arguments:
    ----------
    data: pandas dataframe
        - Data upon which the discriminant is calculated
        - Equivalent to training data
    output_data: pandas dataframe
        - Data that will be projected with computed discriminant
        - Equivalent to testing data
    return_both: bool, optional
        - Default: False
        - If true, returns a tuple of projected data and projected output_data (in that order)

    Returns:
    -------
    pd_projected: pandas dataframe
        - This is output data projected onto discriminant from data
    Return value may be tuple of projected data and projected output_data if return_both = True
    '''

    split_data = split_data_by_class(data, labels = df.iloc[:,-1])

    split_data = {l:df.iloc[:,0:-1] for l, df in split_data.items()}

    # Calculate m dictionary
    m = {int(l):df.shape[0] for l, df in split_data.items()}

    # Calculate classwise means vector
    classwise_mean = {l:[] for l, i in m.items()}

    for l, df in split_data.items(): # Iterate over all classes in split data
        
        for i in range(0, df.shape[1]): # Go over all columns in class data (exclude class)
            # Append mean for that given column to that label's mean vector
            classwise_mean[l].append( np.mean(df.iloc[:,i]) )

    classwise_mean = {l: np.array(m) for l, m in classwise_mean.items()}

    labels = split_data.keys()

    # Calculate S_b and S_w
    # S_w is just equal to (1 - n) * (sum of covariance matrices for each class)
    S_w = np.zeros(shape = (data.shape[1] - 1, data.shape[1] - 1))

    for l in labels: # Add all of the covariance matrices (element-wise)
        S_w = np.add(S_w, np.cov( np.transpose( split_data[l].to_numpy() ) ))

    S_w = np.multiply(len(labels) - 1, S_w) # Multiply by (n - 1) to turn this into a scatter matrix

    # Get the projection vector:
    w = np.matmul(np.linalg.inv(S_w), np.subtract(classwise_mean[1], classwise_mean[0]))
    w = w.reshape(1, data.shape[1] - 1)

    # Project and return the data
    if (return_both):
        return (project_data(w, data), project_data(w, output_data))
    else:
        return (project_data(w, output_data))
