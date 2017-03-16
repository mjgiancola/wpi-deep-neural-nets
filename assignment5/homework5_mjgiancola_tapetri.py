import sys
import numpy as np
from scipy.optimize import check_grad
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

# Compresses weights into 1D weight vector
def pack(W1, W2, b1, b2):
  W1 = np.ravel(W1)
  W2 = np.ravel(W2)
  b1 = b1.flatten()
  b2 = b2.flatten()

  result = np.hstack((W1, W2, b1, b2))

  return result

# Splits 1D weight vector into component weights
def unpack(w, hidden_units):
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

  # print check_grad(lambda w_: _J(w_, grad_batch, grad_label), lambda _w: gradJ(_w, grad_batch, grad_label), w)


# For confirming gradient with check_grad
def _J(w, digits, labels):
  W1, W2, b1, b2 = unpack(w, 30)
  return J(W1, W2, b1, b2, digits, labels)

def J (W1, W2, b1, b2, digits, labels):

  m = digits.shape[0]

  # print ("m is " + str(m))

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

  # print("[feed forward] Dimensions of W1 = " + str(W1.shape))
  # print("[feed forward] Dimensions of W2 = " + str(W2.shape))
  # print("[feed forward] Dimensions of b1 = " + str(b1.shape))
  # print("[feed forward] Dimensions of b2 = " + str(b2.shape))

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

# TODO: db1, dW2, db2, change intermediate value names a, b
def backprop(batch, batch_labels, z1, h1, z2, y_hats, W1, W2, b1, b2):
  """
  Runs backpropagation through the network and returns the gradients for 
  J with respect to W1, W2, b1, and b2.
  """

  # print ("[backprop] Dimension of z1 = " + str(z1.shape))
  # print ("[backprop] Dimension of h1 = " + str(h1.shape))
  # print ("[backprop] Dimension of z2 = " + str(z2.shape))
  # print ("[backprop] Dimension of y_hats = " + str(y_hats.shape))

  y_actuals = batch_labels # np.reshape(batch_labels, (10, 1)) #TODO Fix for multiple instanecs, use batch_labels.shape[1]
  m = batch.shape[0]

  dJdz2 = (y_hats - y_actuals) # num_instances * 10
  dJdh1 = np.dot(dJdz2, W2) # num_instances * 30

  # print ("[backprop] Dimension of dJdz2 = " + str(dJdz2.shape))
  # print ("[backprop] Dimension of dJdh1 = " + str(dJdh1.shape))

  # Equivalently, dJ/dz1 (hadamard product!!)
  g = dJdh1 * relu_prime(z1) # num_instances * 30

  # print ("[backprop] Dimension of g = " + str(g.shape))

  # print ("[backprop] Dimension of batch = " + str(batch.shape))
  # print ("[backprop] Dimension of h1 = " + str(h1.shape))

  # Compute outer product
  dW1 = 1.0 / m * np.outer(g, batch.T) # TODO: ends up being (num_instances*30) * (num_instances*784)
  dW2 = 1.0 / m * np.outer(dJdz2, h1.T) # TODO: ends up being (num_instances*10) * (num_instances*30)

  # Ideally the two above ones are really num_instances * (30 * 784) (3dim) and num_instances * (10 * 30) (3dim)
  # and we simply want to average along the first dimension, meaning we get dW1 being 30 * 784 and dW2 being 10 * 30

  # print ("Dw2 print:")
  # print (np.count_nonzero(dW2))

  # print ("[backprop] Dimension of dW1 = " + str(dW1.shape))
  # print ("[backprop] Dimension of dW2 = " + str(dW2.shape))

  # Gradient is dJ/dz1 * dz1/db1, which is just 1
  # tmp = np.fill_diagonal(np.zeros(h1.shape[1], h1.shape[1]), 1)
  db1 = 1.0 / m * np.sum(np.dot(g, np.identity(b1.size)).T, axis=1, keepdims=True)

  # Similarly, gradient is dJ/dz2 * dz2/db2, which is also 1
  # tmp2 = np.fill_diagonal(np.zeros((b2.shape[0], b2.shape[0])), 1)
  db2 = 1.0 / m * np.sum(np.dot(dJdz2, np.identity(b2.size)).T, axis=1, keepdims=True)

  # print ("[backprop] Dimension of db1 = " + str(db1.shape))
  # print ("[backprop] Dimension of db2 = " + str(db2.shape))

  # print("-----")

  # print("[backprop] Dimensions of W1 = " + str(W1.shape))
  # print("[backprop] Dimensions of W2 = " + str(W2.shape))
  # print("[backprop] Dimensions of b1 = " + str(b1.shape))
  # print("[backprop] Dimensions of b2 = " + str(b2.shape))
  # print("[backprop] Dimensions of dW1 = " + str(dW1.shape))
  # print("[backprop] Dimensions of dW2 = " + str(dW2.shape))
  # print("[backprop] Dimensions of db1 = " + str(db1.shape))
  # print("[backprop] Dimensions of db2 = " + str(db2.shape))


  return dW1, dW2, db1, db2
  

# TODO:
def SGD (trainingData, trainingLabels, hidden_units, learn_rate, batch_size, num_epochs, reg_strength):
  """
  Trains a 3-layer NN with the given hyper parameters and return the weights W1, W2, b1, b2 when done learning.
  """

  # Split into batches (TODO: shuffle?)
  num_of_batches = trainingData.shape[0] / batch_size
  batches = zip(
    np.array_split(trainingData, num_of_batches, axis=0),  # digits
    np.array_split(trainingLabels, num_of_batches, axis=0))  # labels

  # Initialize weight vectors
  (W1, W2, b1, b2) = initialize_weights(hidden_units)
  
  for i in range(0, num_epochs):
    
    # Extract new batch from data
    for (batch_data, batch_labels) in batches:

      print("Current J: " + str(J(W1, W2, b1, b2, trainingDigits, trainingLabels)))

      # Forward propagation
      z1, h1, z2, y_hats = feed_forward(batch_data, W1, W2, b1, b2)

      # Backward propagation
      dW1, dW2, db1, db2 = backprop(batch_data, batch_labels, z1, h1, z2, y_hats, W1, W2, b1, b2)

      if 0:
        print("Before update, " + str(i))
        print("Dimensions of W1 = " + str(W1.shape))
        print("Dimensions of W2 = " + str(W2.shape))
        print("Dimensions of b1 = " + str(b1.shape))
        print("Dimensions of b2 = " + str(b2.shape))
        print("Dimensions of dW1 = " + str(dW1.shape))
        print("Dimensions of dW2 = " + str(dW2.shape))
        print("Dimensions of db1 = " + str(db1.shape))
        print("Dimensions of db2 = " + str(db2.shape))

      # Update weights (TODO: Add regularization)
      W1 = W1 - learn_rate * dW1
      W2 = W2 - learn_rate * dW2
      b1 = b1 - learn_rate * db1
      b2 = b2 - learn_rate * db2

      if 0:
        print("After update, " + str(i))
        print("Dimensions of W1 = " + str(W1.shape))
        print("Dimensions of W2 = " + str(W2.shape))
        print("Dimensions of b1 = " + str(b1.shape))
        print("Dimensions of b2 = " + str(b2.shape))
        print("Dimensions of dW1 = " + str(dW1.shape))
        print("Dimensions of dW2 = " + str(dW2.shape))
        print("Dimensions of db1 = " + str(db1.shape))
        print("Dimensions of db2 = " + str(db2.shape))

      print("Current J: " + str(J(W1, W2, b1, b2, trainingDigits, trainingLabels)))

  return W1, W2, b1, b2

def gradJ(w, batch_data, batch_labels):
  W1, W2, b1, b2 = unpack(w, 30)

  # Forward propagation
  z1, h1, z2, y_hats = feed_forward(batch_data, W1, W2, b1, b2)

  # Backward propagation
  dW1, dW2, db1, db2 = backprop(batch_data, batch_labels, z1, h1, z2, y_hats, W1, W2, b1, b2)

  return pack(dW1, dW2, db1, db2)

def initialize_weights(hidden_units = 30):
  """
  Initializes weight and bias vectors
  """

  w1_abs = 1.0 / np.sqrt(784)
  w2_abs = 1.0 / np.sqrt(hidden_units)

  W1 =  np.random.uniform(-w1_abs,w1_abs,[hidden_units, 784]) # 784 x hidden_units
  W2 = np.random.uniform(-w2_abs,w2_abs,[10, hidden_units]) # hidden_units x 10
  
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
  # W1, W2, b1, b2 = SGD(trainingDigits, trainingLabels, hidden_units = 30, learn_rate = 0.01, batch_size = 1, num_epochs = 1, reg_strength = 0)

  # print (W1)

  print("Relu check")
  a = np.array([[1,-1,2], [-3,0,2]])
  print(a)
  print(relu_prime(a))

  w = pack(W1, W2, b1, b2)

  W1_unpack, W2_unpack, b1_unpack, b2_unpack = unpack(w, 30)

  # print (np.array_equal(W1, W1_unpack))
  # print (np.array_equal(W2, W2_unpack))
  # print (np.array_equal(b1, b1_unpack))
  # print (np.array_equal(b2, b2_unpack))

  grad_batch = trainingDigits[0].T # handles numpy dimension removal
  grad_label = trainingLabels[0].T # handles numpy dimension removal

  print("grad_batch is " + str(grad_batch.shape))
  print("grad_label is " + str(grad_label.shape))
  
  print check_grad(lambda w_: _J(w_, grad_batch, grad_label), lambda _w: gradJ(_w, grad_batch, grad_label), w)

  #W1, W2, b1, b2 = SGD(trainingDigits, trainingLabels, 30, 0.01, 64, 1, 0)
  # print "done"
  # TODO Use check_grad to confirm gradient functions work

  # NOTE: NEW ACCURACY FUNCTION
  # print accuracy(testingDigits, testingLabels, W1, W2, b1, b2)

  # Run gradient descent with learning_rate=0.5, num_iter=325
  # W = gradientDescent(trainingDigits, trainingLabels, W, 0.5, 325)
  
  # print "Loss on Test Set: " + str(J(W, testingDigits, testingLabels))
  # print "Accuracy on Test Set: " + str(accuracy(W, testingDigits, testingLabels))

  # plot_weights_vectors(W)

