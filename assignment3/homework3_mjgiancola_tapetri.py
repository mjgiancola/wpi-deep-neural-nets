# NOTE -- please do NOT put your name(s) in the Python code; instead, name the Python file
# itself to include your WPI username(s).

import sys
import numpy as np
from scipy.optimize import check_grad
from sklearn.linear_model import LogisticRegression

# Performs logistic sigmoid element-wise on matrix x
def sigmoid_array(x):
  return 1.0 / (1.0 + np.exp(-x))

# Original cost function (from Homework 2)
def J (w, faces, labels, alpha = 0.):

    y_hats = w.dot(np.transpose(faces))
    y_actuals = labels
    residue = y_hats - y_actuals
    squared = np.square(residue)
    
    reg = 0.5*alpha * w.dot(w)

    return 0.5 * np.sum(squared) + reg

# Original gradient (from Homework 2)
def gradJ (w, faces, labels, alpha = 0.):
    return faces.T.dot(faces.dot(w) - labels)

# Cross Entropy Loss Function
def J_new (w, faces, labels, alpha = 0.):
    
    y_hats = sigmoid_array(w.dot(np.transpose(faces)))
    y_actuals = labels
    m = y_hats.shape[0]
    ones = np.ones(y_actuals.shape)

    positives = y_actuals*np.log(y_hats) 
    negatives = (ones - y_actuals)*np.log(ones-y_hats)

    regul = 0.5*alpha*np.dot(w.T,w)

    return -1.0/m * np.sum(positives + negatives) + regul

# Gradient of Cross Entropy
def gradJ_new (w, faces, labels, alpha = 0.):

    y_hats = sigmoid_array(w.dot(np.transpose(faces)))
    y_actuals = labels

    m = y_hats.shape[0]
    regul = alpha * w

    result = 1.0/m * np.dot(faces.T, np.transpose((y_hats - y_actuals))) + regul
    return result

# cost and gradient are function pointers for the loss function and respective gradient
# w is the starting weights
# We parameterized learning_rate and tolerance because problems 2 and 3 respond better to different values
# freq determines how often information will be printed (every freq iterations)
def gradientDescent (trainingFaces, trainingLabels, cost, gradient, w, learning_rate, tolerance, freq, alpha = 0.):

    """
    Pick a random starting value for w in R^576 and a small learning rate
    (epis << 1). Then, using the expression for the gradient of the cost
    function, iteratively update w to reduce the cost of J(w). Stop when
    the difference between J over successive training rounds is below some
    tolerance (eg. delta = 0.001).
    """

    # Initialize starting values
    lastJ = np.inf
    currentJ = cost(w, trainingFaces, trainingLabels, alpha) 
    delta = lastJ - currentJ
    num_iter = 1
    
    while (delta > tolerance):
    
        # Problem 2 runs for ~80 iterations
        # Problem 3 runs for ~19000 iterations
        # This allows us to show every iteration for Problem 2,
        # while only showing every 100 iterations for Problem 3
        if (not (num_iter % freq)):
            print("%4d: J = %10f\t||w|| = %5f\tDelta = %5f" % ((num_iter, currentJ, np.linalg.norm(w), delta)))
        
        # Update values
        lastJ = currentJ
        w = w - ( learning_rate * gradient(w, trainingFaces, trainingLabels) )
        currentJ = cost(w, trainingFaces, trainingLabels, alpha)
        delta = abs(lastJ - currentJ)
        num_iter += 1
        
    return w

# Gradient Descent with Squared Error Cost Function
# (Problem 2)
def gd_old(trainingFaces, trainingLabels):
    w = np.zeros(trainingFaces.shape[1])
    learning_rate = 3e-5
    tolerance = 1e-3
    return gradientDescent(trainingFaces, trainingLabels, J, gradJ, w, learning_rate, tolerance, 1)

# Gradient Descent with Cross Entropy Loss Function
# (Problem 3)
def gd_new(trainingFaces, trainingLabels):
    w = np.random.randn(576) / 10
    learning_rate = 0.25 # 2e-3
    tolerance = 8e-7
    return gradientDescent(trainingFaces, trainingLabels, J_new, gradJ_new, w, learning_rate, tolerance, 100)

# Return matrix L such that (trainingFaces*L)^T(trainingFaces*L) = I
def whiten (trainingFaces):

    lam = 1e-3
    cov = np.cov(trainingFaces.T) + lam * np.eye(trainingFaces.shape[1])

    # w - eigenvalues
    # v - eigenvectors    
    w,v = np.linalg.eigh(cov)

    lamd = np.diag(np.float_power(w, (-0.5)))
    phi = v

    L = phi.dot(lamd)
    return L

if __name__ == "__main__":
    # Load data
    if ('trainingFaces' not in globals()):  # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
        trainingFaces = np.load("trainingFaces.npy")
        trainingLabels = np.load("trainingLabels.npy")
        testingFaces = np.load("testingFaces.npy")
        testingLabels = np.load("testingLabels.npy")

    print("#############")
    print("# Problem 2 #")
    print("#############")

    # Whiten training data
    transformedFaces = trainingFaces.dot(whiten(trainingFaces))
    
    # Confirm eigenvalues of covariance matrix almost all close to 1
    cov = np.cov(transformedFaces.T)
    w,v = np.linalg.eigh(cov)
    print(w)
    raw_input("Eigenvalues are almost all close to one.\nPress any key to continue.")

    # Run gradient descent on whitened data
    w_whiten = gd_old(transformedFaces, trainingLabels)
    raw_input("Press any key to continue...")
        
    print("#############")
    print("# Problem 3 #")
    print("#############")
    
    # Initialize weights randomly using Xavier Initialization
    w = np.random.randn(576) / 24
    sys.stdout.write("check_grad on randomly initialized w: ")
    print(check_grad(J_new, gradJ_new, w, trainingFaces, trainingLabels))
    raw_input("Press any key to continue...")

    # Run gradient descent with logistic sigmoid
    w_cross_entropy = gd_new(trainingFaces, trainingLabels)
    
    # Run sklearn LogReg with no bias/regularization, determine optimal weights
    logreg = LogisticRegression(C=1e10, fit_intercept=False)
    logreg.fit(trainingFaces, trainingLabels)
    w_logreg = logreg.coef_[0]

    print("Cost using Sklearn LogReg : %f" % J_new(w_logreg, trainingFaces, trainingLabels))
    print("Cross using Cross Entropy Loss: %f" % J_new(w_cross_entropy, trainingFaces, trainingLabels))

