# NOTE -- please do NOT put your name(s) in the Python code; instead, name the Python file
# itself to include your WPI username(s).

import sys
import numpy as np
from scipy.optimize import check_grad
from sklearn.linear_model import LogisticRegression

###################################
############ TODO #################
###################################
# Update loss function
# Delete code we don't need from hw3

def accuracy(weights, data, labels):

    y_hats = soft_max(np.dot(digits, weights))
    y_hats[np.where(y_hats==np.max(y_hats))] = 1
    a[np.where(a==np.max(a))] = 1

# Performs logistic sigmoid element-wise on matrix x
def sigmoid_array(x):
  return 1.0 / (1.0 + np.exp(-x))

def soft_max(x):
    e = np.exp(x)
    result = e / e.sum(axis = 1)[:, None]
    return result

# Original cost function (from Homework 2)
def J (W, digits, labels):

    # W - 784 * 10
    # digits - 55000 * 784
    m = digits.shape[0]

    y_hats = soft_max(np.dot(digits, W)) # 55000 * 10
    y_actuals = labels # 55000 * 10

    result = -1.0/m * np.sum(np.dot(y_actuals.T, np.log(y_hats))) # 10 * 10

    return result

# Original gradient (from Homework 2)
def gradJ (W, digits, labels):

    m = digits.shape[0]

    # ( y_hats - y_actuals ) * digits
    y_hats = soft_max(np.dot(digits, W)) # 55000 * 10
    y_actuals = labels # 55000 * 10

    result =  1.0/m * np.dot(digits.T, y_hats - y_actuals)

    return result

# cost and gradient are function pointers for the loss function and respective gradient
# w is the starting weights
# We parameterized learning_rate and tolerance because problems 2 and 3 respond better to different values
# freq determines how often information will be printed (every freq iterations)
def gradientDescent (trainingData, trainingLabels, cost, gradient, w, learning_rate, max_iter, freq):

    # Initialize starting values
    lastJ = np.inf
    currentJ = cost(w, trainingData, trainingLabels) 
    delta = lastJ - currentJ
    num_iter = 1
    
    while (num_iter < max_iter):
    
        # Problem 2 runs for ~80 iterations
        # Problem 3 runs for ~19000 iterations
        # This allows us to show every iteration for Problem 2,
        # while only showing every 100 iterations for Problem 3
        if (not (num_iter % freq)):
            print("%4d: J = %10f\t||w|| = %5f\tDelta = %5f" % ((num_iter, currentJ, np.linalg.norm(w), delta)))
        
        # Update values
        lastJ = currentJ
        w = w - ( learning_rate * gradient(w, trainingData, trainingLabels) )
        currentJ = cost(w, trainingData, trainingLabels)
        delta = abs(lastJ - currentJ)
        num_iter += 1
        
    return w

if __name__ == "__main__":
    # Load data
    trainingDigits = np.load("mnist_train_images.npy")
    trainingLabels = np.load("mnist_train_labels.npy")
    testingDigits = np.load("mnist_test_images.npy")
    testingLabels = np.load("mnist_test_labels.npy")
    


    print(trainingDigits.shape)
    print(trainingLabels.shape)
    print(testingDigits.shape)
    print(testingLabels.shape)

    # W = np.random.randn(784, 10) / 10
    W = np.zeros((784,10))
    print(W)

    print("J")
    print(J(W, trainingDigits, trainingLabels))
    
    print("gradJ")
    print(gradJ(W, trainingDigits, trainingLabels))

    w_result = gradientDescent(trainingDigits, trainingLabels, J, gradJ, W, 5, 200 ,1)

    exit()

