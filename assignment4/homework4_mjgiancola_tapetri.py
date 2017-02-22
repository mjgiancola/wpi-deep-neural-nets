# NOTE -- please do NOT put your name(s) in the Python code; instead, name the Python file
# itself to include your WPI username(s).

import sys
import numpy as np
from scipy.optimize import check_grad
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def accuracy(weights, data, labels):

    y_hats = soft_max(np.dot(digits, weights))
    y_actuals = labels
    m = len(labels)
    
    return sum( np.argmax(y_hats, axis=1) == labels ) / m

def soft_max(x):
    e = np.exp(x)
    result = e / e.sum(axis = 1)[:, None]
    return result

def J (W, digits, labels):

    # W - 784 * 10
    # digits - 55000 * 784
    m = digits.shape[0]

    y_hats = soft_max(np.dot(digits, W)) # 55000 * 10
    y_actuals = labels # 55000 * 10

    result = -1.0/m * np.sum(np.multiply(y_actuals, np.log(y_hats))) # 10 * 10

    return result

def gradJ (W, digits, labels):

    m = digits.shape[0]

    # ( y_hats - y_actuals ) * digits
    y_hats = soft_max(np.dot(digits, W)) # 55000 * 10
    y_actuals = labels # 55000 * 10

    result =  1.0/m * np.dot(digits.T, y_hats - y_actuals)

    return result

def gradientDescent (trainingData, trainingLabels, w, learning_rate, max_iter):

    # Initialize starting values
    lastJ = np.inf
    currentJ = J(w, trainingData, trainingLabels) 
    delta = lastJ - currentJ
    num_iter = 1
    
    while (num_iter < max_iter):

        print("%4d: J = %10f\t||w|| = %5f\tDelta = %5f" % ((num_iter, currentJ, np.linalg.norm(w), delta)))
        
        # Update values
        lastJ = currentJ
        w = w - ( learning_rate * gradJ(w, trainingData, trainingLabels) )
        currentJ = J(w, trainingData, trainingLabels)
        delta = abs(lastJ - currentJ)
        num_iter += 1
        
    return w

if __name__ == "__main__":
    # Load data
    trainingDigits = np.load("mnist_train_images.npy")
    trainingLabels = np.load("mnist_train_labels.npy")
    testingDigits = np.load("mnist_test_images.npy")
    testingLabels = np.load("mnist_test_labels.npy")
    
    W = np.zeros((784,10))

    w_result = gradientDescent(trainingDigits, trainingLabels, W, 0.5, 200)

    exit()

