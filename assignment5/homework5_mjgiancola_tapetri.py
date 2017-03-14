import sys
import numpy as np
from scipy.optimize import check_grad
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# TODO Update
def accuracy(weights, digits, labels):

  y_hats = soft_max(np.dot(digits, weights))
  y_actuals = labels
  
  return np.mean(np.argmax(y_hats, axis=1) == np.argmax(y_actuals, axis=1))

def plot_weights_vectors(w):
    
  plt.figure(1)
  for i in xrange(1,11):
      plt.subplot(1,10,i)
      plt.imshow(w.T[i-1].reshape((28,28)), cmap='gray')
      plt.axis('off')
  plt.show()

def relu(x):
  return [max(0, elt) for elt in x]

def relu_prime(x):
  return [1 if elt > 0 else 0 for elt in x]

def soft_max(x):
  e = np.exp(x)
  result = e / e.sum(axis = 1)[:, None]
  return result

# Computes y_hats for current weights/biases
def feed_forward(W1, b1, W2, b2, digits):

  # 55000 * 10
  z1 = np.dot(digits, W1) + b1
  h1 = relu(z1)

  # 55000 * ? (10 I think)
  z2 = np.dot(h1, W2) + b2
  y_hats = softmax(z2)

  return y_hats

def J (W1, b1, W2, b2, digits, labels):

  # W - 784 * 10
  # digits - 55000 * 784
  m = digits.shape[0]

  # 55000 * 10
  y_hats = feed_forward(W1, b1, W2, b2)
  y_actuals = labels

  result = -1.0/m * np.sum(np.multiply(y_actuals, np.log(y_hats)))

  return result

# TODO Fix
def gradJW1 (W1, b1, W2, b2, digits, labels):

  x = digits

  # 55000 * 10
  y_hats = feed_forward(W1, b2, W2, b2)
  y_actuals = labels
  a = np.dot((y_hats - y_actuals), W2)

  z1 = np.dot(digits, W1) + b1
  b = relu_prime(z1)

  g = (a * b).T

  # Compute outer product
  result =  np.dot(g, x.T)

  return result

# TODO Make stochastic
def SGD (trainingData, trainingLabels, w, learning_rate, num_epochs):

    # Initialize starting values
    lastJ = np.inf
    currentJ = J(w, trainingData, trainingLabels) 
    delta = lastJ - currentJ
    
    for num_iter in range(1, num_epochs):

        print("%4d: J = %10f\t||w|| = %5f\tDelta = %5f" % ((num_iter, currentJ, np.linalg.norm(w), delta)))
        
        # Update values
        lastJ = currentJ
        w = w - ( learning_rate * gradJ(w, trainingData, trainingLabels) )
        currentJ = J(w, trainingData, trainingLabels)
        delta = abs(lastJ - currentJ)
        
    return w

def initialize_weights(hidden_units = 30):

    w1_abs = 1.0 / sqrt(784)
    w2_abs = 1.0 / sqrt(30)

    W_1 =  np.random.uniform(-w1_abs,w1_abs,[784, hidden_units]) # 784 x hidden_units
    W_2 = np.random.uniform(-w2_abs,w2_abs,[hidden_units, 10]) # hidden_units x 10
    b_1 = 0.01 * np.ones((hidden_units,1)) # hidden_units x 1
    b_2 = 0.01 * np.ones((10,1)) # 10 x 1

    return W_1, W_2, b_1, b_2

if __name__ == "__main__":
   
    # Load data
    trainingDigits = np.load("mnist_train_images.npy")
    trainingLabels = np.load("mnist_train_labels.npy")
    validationDigits = np.load("mnist_validation_images.npy")
    validationLabels = np.load("mnist_validation_labels.npy")
    testingDigits = np.load("mnist_test_images.npy")
    testingLabels = np.load("mnist_test_labels.npy")
    

    # Hyper parameters
    num_hidden_units = [30,40,50]
    learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    minibatch_size = [16, 32, 64, 128, 256]
    num_epochs = [100,1000,10000] # ?
    reg_strength = [0.01, 0.1, 0.2, 0.5]

    # For each set of hyper param settings
    #    * train NN on training set
    #    * validate NN on validation set
    # Report final accuracy on test set

    hidden_units = 30

    # Initialize weight vectors with all zeros
    W_1 =  np.random.uniform(5,10,[784, 30]) # 784 x hidden_units
    W_2 = np.random.uniform(5,10,[30, 10]) # hidden_units x 10
    b_1 = 0.01 * np.ones((30,1)) # hidden_units x 1
    b_2 = 0.01 * np.ones((10,1)) # 10 x 1



    # Run gradient descent with learning_rate=0.5, num_iter=325
    W = gradientDescent(trainingDigits, trainingLabels, W, 0.5, 325)
    
    print "Loss on Test Set: " + str(J(W, testingDigits, testingLabels))
    print "Accuracy on Test Set: " + str(accuracy(W, testingDigits, testingLabels))

    plot_weights_vectors(W)

