# NOTE -- please do NOT put your name(s) in the Python code; instead, name the Python file
# itself to include your WPI username(s).

import cv2  # Uncomment if you have OpenCV and want to run the real-time demo
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
    print(residue)
    squared = np.square(residue)
    
    reg = 0.5*alpha * w.dot(w)

    return 0.5 * np.sum(squared) + reg

# Original gradient (from Homework 2)
def gradJ (w, faces, labels, alpha = 0.):
    #     Gradient = x^T * (x*w - y)
    gradient = faces.T.dot(faces.dot(w) - labels)

    return gradient

# Cross Entropy Loss Function
def J_new (w, faces, labels, alpha = 0.):
    
    #print(w.dot(np.transpose(faces)))
    
    y_hats = sigmoid_array(w.dot(np.transpose(faces)))
    # print (y_hats)
    y_actuals = labels
    m = y_hats.shape[0]
    ones = np.ones(y_actuals.shape)

    positives = y_actuals*np.log(y_hats) 
    negatives = (ones - y_actuals)*np.log(ones-y_hats+1e-8)

    regul = 0.5*alpha*np.dot(w.T,w)

    return -1.0/m * np.sum(positives + negatives) + regul

# Gradient of Cross Entropy
def gradJ_new (w, faces, labels, alpha = 0.):


    y_hats = sigmoid_array(w.dot(np.transpose(faces)))
    # print("in gradJ_new: y_hats")
    # print(y_hats)
    y_actuals = labels
    # print("in gradJ_new: y_actuals")
    # print(y_actuals)
    m = y_hats.shape[0]
    # print("in gradJ_new: m")
    # print(m)
    regul = alpha * w
    # print("in gradJ_new: regul")
    # print(regul)

    # print("in gradJ_new: y_hats-y_actuals")
    # print(np.transpose((y_hats - y_actuals)))

    # print("in gradJ_new: faces")
    # print(faces)

    result = 1.0/m * np.dot(faces.T, np.transpose((y_hats - y_actuals))) + regul
    # result = -1.0/m * np.sum(y_actuals - y_hats)

    # print("in gradJ_new: gradient")
    # print(result.shape)
    # print(result)

    return result

# cost and gradient are function pointers for the loss function and respective gradient
def gradientDescent (trainingFaces, trainingLabels, cost, gradient, alpha = 0.):

    """
    Pick a random starting value for w in R^576 and a small learning rate
    (epis << 1). Then, using the expression for the gradient of the cost
    function, iteratively update w to reduce the cost of J(w). Stop when
    the difference between J over successive training rounds is below some
    tolerance (eg. delta = 0.001).
    """

    # parameters for problem 3
    learning_rate = 3e-3
    tolerance = 1e-4

    # parameters for problem 2
    # learning_rate = 3e-5
    # tolerance = 1e-3

    #w = np.zeros(trainingFaces.shape[1])  # Or set to random vector
    w = np.random.choice(5, 576) 

    lastJ = np.inf
    currentJ =  cost(w, trainingFaces, trainingLabels, alpha) 
    delta = lastJ - currentJ

    num_iter = 1
    
    # print lastJ
    # print currentJ
    # print(w)
    while (delta > tolerance):
    
        if (not (num_iter % 100)): 
            print("%2d: J = %10f\t||w|| = %5f\tDelta = %5f" % ((num_iter, currentJ, np.linalg.norm(w), delta)))
        
    
        lastJ = currentJ
        w = w - ( learning_rate * gradient(w, trainingFaces, trainingLabels) )

        # print("-----")
        # print(gradient(w, trainingFaces, trainingLabels))
        # print(w)
        # print("-----")
        currentJ = cost(w, trainingFaces, trainingLabels, alpha)
        delta = abs(lastJ - currentJ)

        # print("new J")
        # print(currentJ)
        
        num_iter += 1
        
    return w

# Unregularized Gradient Descent with Squared Error Cost Function
# (Problem 2)
def gd_unreg (trainingFaces, trainingLabels):
    return gradientDescent(trainingFaces, trainingLabels, J, gradJ)

# Regularized Gradient Descent with Cross Entropy Loss Function
# (Problem 3)
def gd_reg (trainingFaces, trainingLabels):
    alpha = 0
    return gradientDescent(trainingFaces, trainingLabels, J_new, gradJ_new, alpha)

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

    if(0):

        # Whiten training data
        transformedFaces = trainingFaces.dot(whiten(trainingFaces))
        
        # Confirm eigenvalues of covariance matrix almost all close to 1
        cov = np.cov(transformedFaces.T)
        w,v = np.linalg.eigh(cov)
        print(w)
        raw_input("Eigenvalues are almost all close to one.\nPress any key to continue.")

        # Run (unregularized) gradient descent on whitened data
        w_whiten = gd_unreg(transformedFaces, trainingLabels)
        
    # TODO Problem 3
    
    w = np.random.choice(5, 576) 
    
    print (check_grad(J_new, gradJ_new, w, trainingFaces, trainingLabels))
    raw_input("Ideally that would have been really low...")

    w_cross_entropy = gd_reg(trainingFaces, trainingLabels)
    result_cross_entropy = J_new (w_cross_entropy, testingFaces, testingLabels, alpha = 0.)

    logreg = LogisticRegression(C=1e10, fit_intercept=False)
    logreg.fit(trainingFaces, trainingLabels)

    w_logreg = logreg.coef_[0]

    result_logreg = J_new (w_logreg, testingFaces, testingLabels, alpha = 0.)

    print("Logreg:")
    print(result_logreg)
    print("Cross Entropy:")
    print(result_cross_entropy)


