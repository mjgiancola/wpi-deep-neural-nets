import sys
import numpy as np
from scipy.optimize import check_grad
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
  return np.maximum(x, 0)

# TODO: Use numpy to optimize
def relu_prime(x):
  return [1 if elt > 0 else 0 for elt in x]

def soft_max(x):
    row_max = np.max(x, axis = 1,keepdims=True)
    max_removed = x - row_max
    e_x = np.exp(max_removed)
    return e_x / e_x.sum(axis = 1, keepdims=True) 

return z1, h1, z2, y_hats

def J (W1, b1, W2, b2, digits, labels):

  # W - 784 * 10
  # digits - 55000 * 784
  m = digits.shape[0]

  # 55000 * 10
  y_hats = feed_forward(W1, b1, W2, b2, digits)
  y_actuals = labels

  result = -1.0/m * np.sum(np.multiply(y_actuals, np.log(y_hats)))

  return result

def feed_forward(batch, W1, b1, W2, b2):
  """
  Runs the given batch through the network defined by the given weights
  and return the intermediate values z1, h1, z2 as well as the final y_hats
  """

  # digits is num_instances (varies) * 784
  # W1 is 784 * 30
  # z1 should be num_instances * 30

  # 55000 * 30
  z1_no_bias = np.dot(digits, W1)
  z1 =  z1_no_bias + b1.T
  h1 = relu(z1)

  # 55000 * 10 
  z2_no_bias = np.dot(h1, W2)
  z2 = z2_no_bias + b2.T
  y_hats = soft_max(z2)

# TODO: db1, dW2, db2, change intermediate value names a, b
def backprop(batch, batch_labels, z1, h1, z2, y_hats, W1, b1, W2, b2):
  """
  Runs backpropagation through the network and returns the gradients for 
  J with respect to W1, b1, W2, and b2.
  """

  # 55000 * 10
  a = np.dot((y_hats - batch_labels), W2)

  z1 = np.dot(batch, W1) + b1
  b = relu_prime(z1)

  g = (a * b).T

  # Compute outer product
  dW1 =  np.dot(g, batch.T)

  return dW1, db1, dW2, db2
  

# TODO:
def SGD (trainingData, trainingLabels, hidden_units, learn_rate, batch_size, num_epochs, reg_strength):
  """
  Trains a 3-layer NN with the given hyper parameters and return the weights W1, b1, W2, b2 when done learning.
  """

    # Initialize weight vectors
    (W1, W2, b1, b2) = initialize_weights(hidden_units)

    # Initialize starting values
    lastJ = np.inf
    currentJ = J(W1, b1, W2, b2, trainingDigits, trainingLabels)
    delta = lastJ - currentJ
    
    for epoch in range(1, num_epochs):

      # Extract new batch from data / Loop over all batches once?

      # Compute J using feedforward

      # Update weights using backprop values * learning rate


    return W1, b1, W2, b2


def initialize_weights(hidden_units = 30):

    w1_abs = 1.0 / np.sqrt(784)
    w2_abs = 1.0 / np.sqrt(30)

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

    hidden_units = 30 # default for now

    # Initialize weight vectors
    (W_1, W_2, b_1, b_2) = initialize_weights(hidden_units)

    print("Initial cost for initialized weights, J = " + str(J(W_1, b_1, W_2, b_2, trainingDigits, trainingLabels)))

    # Run gradient descent with learning_rate=0.5, num_iter=325
    # W = gradientDescent(trainingDigits, trainingLabels, W, 0.5, 325)
    
    # print "Loss on Test Set: " + str(J(W, testingDigits, testingLabels))
    # print "Accuracy on Test Set: " + str(accuracy(W, testingDigits, testingLabels))

    # plot_weights_vectors(W)

