import sys
import numpy as np
from numpy.linalg import norm
from scipy.optimize import check_grad

########## CHECK_GRAD UTILITY FUNCTIONS ##########

def pack(W1, W2, b1, b2):
  """
  Compresses weights/biases into 1D weight vector
  """

  W1 = np.ravel(W1)
  W2 = np.ravel(W2)
  b1 = b1.flatten()
  b2 = b2.flatten()

  return np.hstack((W1, W2, b1, b2))

def unpack(w, hidden_units):
  """
  Splits 1D weight vector into component weights
  """

  a = 784 * hidden_units
  b = a + 300
  c = b + 30
  d = c + 10

  W1 = w[:a]
  W2 = w[a:b]
  b1 = w[b:c]
  b2 = w[c:d]

  W1 = np.reshape(W1, (hidden_units, 784))
  W2 = np.reshape(W2, (10, hidden_units))
  b1 = np.reshape(b1, (hidden_units, 1))
  b2 = np.reshape(b2, (10, 1))

  return W1, W2, b1, b2

def _J(w, digits, labels):
  """
  Wrapper of cost function for confirming gradient with check_grad
  """

  W1, W2, b1, b2 = unpack(w, 30)
  return J(W1, W2, b1, b2, digits, labels)

def gradJ(w, batch_data, batch_labels):
  """
  Gradient function (for the purposes of testing with check_grad)
  """

  W1, W2, b1, b2 = unpack(w, 30)

  # Forward propagation
  z1, h1, z2, y_hats = feed_forward(batch_data, W1, W2, b1, b2)

  # Backward propagation
  dW1, dW2, db1, db2 = backprop(batch_data, batch_labels, z1, h1, z2, y_hats, W1, W2, b1, b2)

  return pack(dW1, dW2, db1, db2)

########## ACTIVATION FUNCTIONS ##########

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

########## MAIN NEURAL NETWORK CODE ##########

def accuracy(W1, W2, b1, b2, digits, labels):

  _, _, _, y_hats = feed_forward(digits, W1, W2, b1, b2)
  y_actuals = labels
  
  return np.mean(np.argmax(y_hats, axis=1) == np.argmax(y_actuals, axis=1))

def J (W1, W2, b1, b2, digits, labels, alpha=0.):

  m = digits.shape[0]

  _, _, _, y_hats = feed_forward(digits, W1, W2, b1, b2)
  y_actuals = labels
  regul = 0.5 * alpha * (norm(W1) + norm(W2))
  result = -1.0/m * np.sum(np.multiply(y_actuals, np.log(y_hats))) + regul

  return result

def feed_forward(batch, W1, W2, b1, b2):
  """
  Runs the given batch through the network defined by the given weights
  and return the intermediate values z1, h1, z2 as well as the final y_hats
  """

  # batch is num_instances (varies) * 784
  # W1 is 784 * num_hidden_units (varies)
  # z1 (and h1) should be num_instances * num_hidden_units

  z1_no_bias = np.dot(batch, W1.T)
  z1 =  z1_no_bias + b1.T
  h1 = relu(z1)

  # h1 is num_instances * num_hidden_units
  # W2 is num_hidden_units * 10
  # z2 (and y_hats) should be num_instances * 10

  z2_no_bias = np.dot(h1, W2.T)
  z2 = z2_no_bias + b2.T
  y_hats = soft_max(z2)

  return z1, h1, z2, y_hats

def backprop(batch, batch_labels, z1, h1, z2, y_hats, W1, W2, b1, b2, alpha=0.):
  """
  Runs backpropagation through the network and returns the gradients for 
  J with respect to W1, W2, b1, and b2.
  """

  y_actuals = batch_labels
  m = batch.shape[0]

  dJdz2 = (y_hats - y_actuals) # num_instances * 10
  dJdh1 = np.dot(dJdz2, W2) # num_instances * 3

  # Equivalently, dJ/dz1 (Hadamard Product)
  g = (dJdh1 * relu_prime(z1)).T # num_instances * 30

  # Compute outer product
  dW1 = 1.0 / m * np.dot(g, batch) + alpha*W1
  dW2 = 1.0 / m * np.dot(dJdz2.T, h1) + alpha*W2

  # Gradient is dJ/dz1 * dz1/db1, which is just 1
  db1 = 1.0 / m * np.sum(np.dot(g.T, np.identity(b1.size)).T, axis=1, keepdims=True)

  # Similarly, gradient is dJ/dz2 * dz2/db2, which is also 1
  db2 = 1.0 / m * np.sum(np.dot(dJdz2, np.identity(b2.size)).T, axis=1, keepdims=True)

  return dW1, dW2, db1, db2
  
def SGD (trainingData, trainingLabels, hidden_units, learn_rate, batch_size, num_epochs, reg_strength):
  """
  Trains a 3-layer NN with the given hyper parameters and return the weights W1, W2, b1, b2 when done learning.
  """

  # Split into batches
  num_samples = trainingData.shape[0]
  num_of_batches = num_samples / batch_size

  # Initialize weight vectors
  (W1, W2, b1, b2) = initialize_weights(hidden_units)
  
  for i in range(0, num_epochs):

    for j in range(0, num_of_batches):
      samples = np.random.choice(num_samples, batch_size)

      batch_data = trainingData[samples]
      batch_labels = trainingLabels[samples]

      # Forward propagation
      z1, h1, z2, y_hats = feed_forward(batch_data, W1, W2, b1, b2)

      # Backward propagation
      dW1, dW2, db1, db2 = backprop(batch_data, batch_labels, z1, h1, z2, y_hats, W1, W2, b1, b2, reg_strength)

      # Update weights
      W1 = W1 - learn_rate * dW1
      W2 = W2 - learn_rate * dW2
      b1 = b1 - learn_rate * db1
      b2 = b2 - learn_rate * db2

    print("Epoch " + str(i) + ", J = " + str(J(W1, W2, b1, b2, trainingDigits, trainingLabels, reg_strength)))

  return W1, W2, b1, b2

def initialize_weights(hidden_units = 30):
  """
  Initializes weight and bias vectors
  """

  w1_abs = 1.0 / np.sqrt(784)
  w2_abs = 1.0 / np.sqrt(hidden_units)

  W1 =  np.random.uniform(-w1_abs,w1_abs,[hidden_units, 784])
  W2 = np.random.uniform(-w2_abs,w2_abs,[10, hidden_units])
  
  b1 = 0.01 * np.ones((hidden_units,1))
  b2 = 0.01 * np.ones((10,1))

  return W1, W2, b1, b2

def findBestHyperparameters(trainingDigits, trainingLabels, validationDigits, validationLabels):
  """
  Loops over selected values to determine best hyperparameter settings
  """

  # Hyperparameter options
  num_hidden_units = [30,40,50]
  learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
  minibatch_size = [16, 32, 64, 128, 256]
  num_epochs = [100,1000,10000] # ?
  reg_strength = [0.01, 0.1, 0.2, 0.5]

  chosen_params = {}

  # For each set of hyper param settings
  #    * train NN on training set
  #    * validate NN on validation set
  
  # num_epochs and regularization are fixed for all tests (Total: 11 tests)

  # Fix hidden_units, batch_size, test all options of learn_rate
  # (6 tests)
  print "Optimizing Learning Rate..."
  scores = []
  for lr in learning_rate:
    W1, W2, b1, b2 = SGD(trainingDigits, trainingLabels, hidden_units=30, learn_rate=lr, batch_size=64, num_epochs=30, reg_strength=0)
    scores.append(J(W1,W2,b1,b2, validationDigits, validationLabels))
  chosen_params['lr'] = [y for (x,y) in sorted(zip(scores, learning_rate))][0]
  print "Best Learning Rate: " + str(chosen_params['lr'])

  # Fix learning_rate to best, keep hidden_units fixed, test all options of minibatch_size
  # (5 tests)
  print "Optimizing Minibatch Size..."
  scores = []
  for mb in minibatch_size:
    W1, W2, b1, b2 = SGD(trainingDigits, trainingLabels, hidden_units=30, learn_rate=chosen_params['lr'], batch_size=mb, num_epochs=30, reg_strength=0)
    scores.append(J(W1,W2,b1,b2, validationDigits, validationLabels))
  chosen_params['mb'] = [y for (x,y) in sorted(zip(scores, minibatch_size))][0]
  print "Best Minibatch Size: " + str(chosen_params['mb'])

  # Fix learing_rate and batch_size to best, test all options of hidden_units
  # (3 tests)
  print "Optimizing Number of Hidden Units..."
  scores = []
  for hu in num_hidden_units:
    W1, W2, b1, b2 = SGD(trainingDigits, trainingLabels, hidden_units=hu, learn_rate=chosen_params['lr'], batch_size=chosen_params['mb'], num_epochs=30, reg_strength=0)
    scores.append(J(W1,W2,b1,b2, validationDigits, validationLabels))
  chosen_params['hu'] = [y for (x,y) in sorted(zip(scores, num_hidden_units))][0]
  print "Best Number of Hidden Units: " + str(chosen_params['hu'])

  chosen_params['ep'] = 100 # More epochs generally improves performance, and 100 doesn't take *too* long
  chosen_params['rg'] = 0   # Regularization didn't have a significant effect on our performance

  print "\nFinal Hyperparameter Choices:"
  print "Number of Hidden Units: " + str(chosen_params['hu'])
  print "Learning Rate: " + str(chosen_params['lr'])
  print "Minibatch Size: " + str(chosen_params['mb'])
  print "Number of Epochs: 100"
  print "Regularization Strength: 0\n"

  return chosen_params

if __name__ == "__main__":
   
  # Load data
  trainingDigits = np.load("mnist_train_images.npy")
  trainingLabels = np.load("mnist_train_labels.npy")
  validationDigits = np.load("mnist_validation_images.npy")
  validationLabels = np.load("mnist_validation_labels.npy")
  testingDigits = np.load("mnist_test_images.npy")
  testingLabels = np.load("mnist_test_labels.npy")

  (W1, W2, b1, b2) = initialize_weights()
  print("Cost for initialized weights, J = " + str(J(W1, W2, b1, b2, trainingDigits, trainingLabels)))

  # Confirm gradient expression is correct
  w = pack(W1, W2, b1, b2)
  grad_batch = trainingDigits[0,:,None].T # handles numpy dimension removal
  grad_label = trainingLabels[0,:,None].T # handles numpy dimension removal
  print "check_grad: " + str(check_grad(lambda w_: _J(w_, grad_batch, grad_label), lambda _w: gradJ(_w, grad_batch, grad_label), w))

  params = findBestHyperparameters(trainingDigits, trainingLabels, validationDigits, validationLabels)

  # Run stochastic gradient descent with optimal hyperparameters on test set
  print "Training network with optimial hyperparameters..."
  W1, W2, b1, b2 = SGD(trainingDigits, trainingLabels, \
                       hidden_units = params['hu'], learn_rate = params['lr'], batch_size = params['mb'], num_epochs = params['ep'], reg_strength = params['rg'])

  print "\nResults on Testing Set"
  print "Accuracy: " + str(accuracy(W1, W2, b1, b2, testingDigits, testingLabels))
  print "Final (unregularized) Cost: " + str(J(W1, W2, b1, b2, testingDigits, testingLabels))
