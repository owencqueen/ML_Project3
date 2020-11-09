# ML_Project3
COSC 522 Project 3: Back Propagation Neural Networks (BPNN)
Owen Queen

The report for this project is in .

![digits](https://github.com/owencqueen/ML_Project3/blob/main/mnist_pictures.png)

## Running the code
Please see the Jupyter Notebook [`run_project3.ipynb`](https://github.com/owencqueen/ML_Project3/blob/main/run_project3.ipynb) in order to run the code. This file will contain any information about running code that is not in the Notebook file.

### Before running:
You need to create several directories in the task2 directory. First, create a directory named `pca_mnist` with child directories named `knn_pca`, `train`, `test`, and `validation`. Next, create a directory named `knn_mnist`. Then you should be able to run the code. 

## Dependencies:
The programs in this repository use the following Python third-party modules:
1. matplotlib
2. numpy

All other modules used are derived from standard libraries included with most Python installations.

## Structure of Repository

1. task2: contains code/files related to Task 2, mainly the bonus question
2. README.md: information about the repo
2. dim_reduce.py: contains functions for dimensionality reduction.
3. load_XOR.py: loads in the XOR datasets
4. mnist_loader.py: contains functions to load in MNIST
5. mylearn.py: contains kNN and MPP code
6. network_oq.py: contains all neural network code
7. run_project3.ipynb: notebook to run code for the project
8. util.py: Various utilities for use by a variety of files

## Acknowledgements/ Citations:
The `network_oq.py` file is a modification of a program in Michael Nielsen's "Neural Networks and Deep Learning" textbook. The code that I used was translated by Michal Daniel Dobrzanski into Python 3. The source repository can be found [here](https://github.com/MichalDanielDobrzanski/DeepLearningPython35).

`util.py` and `mylearn.py` contain code that was written by Dr. Hairong Qi at the University of Tennessee for use in this project.
