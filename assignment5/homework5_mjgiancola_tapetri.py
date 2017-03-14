import sys
import numpy as np
from scipy.optimize import check_grad
import matplotlib.pyplot as plt

def accuracy(weights, digits, labels):

  _, _, _, y_hats = feed_forward(digits, W1, W2, b1, b2)
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

def relu_prime(x):
  result = np.zeros(x.shape)
  result[np.nonzero(x>=0)] = 1 # To satisfy check_grad, relu' is 1 at 0
  return result

def soft_max(x):
    row_max = np.max(x, axis = 1,keepdims=True)
    max_removed = x - row_max
    e_x = np.exp(max_removed)
    return e_x / e_x.sum(axis = 1, keepdims=True) 

def J (W1, W2, b1, b2, digits, labels):

  m = digits.shape[0]

  _, _, _, y_hats = feed_forward(digits, W1, W2, b1, b2)
  y_actuals = labels

  result = -1.0/m * np.sum(np.multiply(y_actuals, np.log(y_hats)))

  return result

def feed_forward(batch, W1, W2, b1, b2):
  """
  Runs the given batch through the network defined by the given weights
  and return the intermediate values z1, h1, z2 as well as the final y_hats
  """

  # batch is num_instances (varies) * 784
  # W1 is 784 * num_hidden_units (varies)
  # z1 (and h1) should be num_instances * num_hidden_units

  z1_no_bias = np.dot(batch, W1)
  z1 =  z1_no_bias + b1.T
  h1 = relu(z1)

  # h1 is num_instances * num_hidden_units
  # W2 is num_hidden_units * 10
  # z2 (and y_hats) should be num_instances * 10

  z2_no_bias = np.dot(h1, W2)
  z2 = z2_no_bias + b2.T
  y_hats = soft_max(z2)

  return z1, h1, z2, y_hats

# TODO: db1, dW2, db2, change intermediate value names a, b
def backprop(batch, batch_labels, z1, h1, z2, y_hats, W1, W2, b1, b2):
  """
  Runs backpropagation through the network and returns the gradients for 
  J with respect to W1, W2, b1, and b2.
  """

  y_actuals = batch_labels

  dJdz2 = (y_hats - y_actuals)
  dJdh1 = np.dot(dJdz2, W2)

  # Equivalently, dJ/dz1
  g = (dJdh1 * relu_prime(z1)).T

  # Compute outer product
  dW1 = np.dot(g, batch.T)
  dW2 = np.dot(dJdz2, h1)

  # Gradient is dJ/dz1 * dz1/db1, which is just 1
  db1 = np.dot(g, np.fill_diagonal(np.zeros((h1.shape[1], h1.shape[1])), 1))

  # Similarly, gradient is dJ/dz2 * dz2/db2, which is also 1
  db2 = np.dot(dJdz2, np.fill_diagonal(np.zeros((b2.shape[0], b2.shape[0])), 1))

  return dW1, dW2, db1, db2
  

# TODO:
def SGD (trainingData, trainingLabels, hidden_units, learn_rate, batch_size, num_epochs, reg_strength):
  """
  Trains a 3-layer NN with the given hyper parameters and return the weights W1, W2, b1, b2 when done learning.
  """

  # Split into batches (TODO: shuffle?)
  num_of_batches = trainingData.shape[0] / batch_size
  batches = zip(
    np.array_split(trainingData, nb_of_batches, axis=0),  # digits
    np.array_split(trainigLabels, nb_of_batches, axis=0))  # labels

  # Initialize weight vectors
  (W1, W2, b1, b2) = initialize_weights(hidden_units)
  
  for i in range(1, num_epochs):
    
    # Extract new batch from data
    for (batch_data, batch_labels) in batches:

      # Forward propagation
      z1, h1, z2, y_hats = feedforward(batch_data, W1, W2, b1, b2)

      # Backward propagation
      dW1, dW2, db1, db2 = backprop(batch_data, batch_labels, z1, h1, z2, y_hats, W1, W2, b1, b2)

      # Update weights (TODO: Add regularization)
      W1 = W1 - learn_rate * dW1
      W2 = W2 - learn_rate * dW2
      b1 = b1 - learn_rate * db1
      b2 = b2 - learn_rate * db2

    # print info
    if i % 10 == 0:
      print("Current J: " + str(J(W1, W2, b1, b2, trainingDigits, trainingLabels)))

  return W1, W2, b1, b2


def initialize_weights(hidden_units = 30):
  """
  Initializes weight and bias vectors
  """

  w1_abs = 1.0 / np.sqrt(784)
  w2_abs = 1.0 / np.sqrt(30)

  W1 =  np.random.uniform(-w1_abs,w1_abs,[784, hidden_units]) # 784 x hidden_units
  W2 = np.random.uniform(-w2_abs,w2_abs,[hidden_units, 10]) # hidden_units x 10
  b1 = 0.01 * np.ones((hidden_units,1)) # hidden_units x 1
  b2 = 0.01 * np.ones((10,1)) # 10 x 1

  return W1, W2, b1, b2

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

  # Initialize weight vectors
  (W1, W2, b1, b2) = initialize_weights()

  print("Initial cost for initialized weights, J = " + str(J(W1, W2, b1, b2, trainingDigits, trainingLabels)))

  # TODO Use check_grad to confirm gradient functions work

  # NOTE: NEW ACCURACY FUNCTION
  # print accuracy(testingDigits, testingLabels, W1, W2, b1, b2)

  # Run gradient descent with learning_rate=0.5, num_iter=325
  # W = gradientDescent(trainingDigits, trainingLabels, W, 0.5, 325)
  
  # print "Loss on Test Set: " + str(J(W, testingDigits, testingLabels))
  # print "Accuracy on Test Set: " + str(accuracy(W, testingDigits, testingLabels))

  # plot_weights_vectors(W)

