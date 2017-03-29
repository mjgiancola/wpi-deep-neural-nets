import tensorflow as tf
import numpy as np

# Outputs string to log file and stdout
def _print(string):
  logfile.write(string + "\n") # Add newline to be consistent with standard print function
  print string

# Import MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create logfile
import datetime, os
if not os.path.exists("logs/"): os.makedirs("logs/")
logfile = open("logs/" + str(datetime.datetime.now()).replace(' ', '_')[:19] + ".log", 'w')

""" TODO Delete
* Two hidden layers, varying number of units
* With and without Dropout 0.5 or 1
* Weight initialization 
  * (std: 1/sqrt(previs # units) for now)
  * unit 
* Try three different optimizers (SGD, Momentum, Adam)
* batch size fixed 128
* num epochs fixed 1000
* activation functions
  * try relu on all for now (experiment with tanh later)

* SGD
* Momentum,
* Adam with learning rate schedule

We need:
* setup model (num_hidden_units, dr)
* train model (batch size, num epochs)

"""

# Fixed
NUM_EPOCHS = 100
BATCH_SIZE = 128
NUM_PIXELS = 784
OUTPUT_UNITS = 10

"""
When this is set to True, the script optimizes a given set
of hyperparameters, and verifies accuracy using the
validation set. When set to False, the script trains using
the global values given below, and does a final test of
accuracy when run on the testing data
"""
optimize_hyperparameters = False

# Optimized Global Parameters
global_learning_rate = 0.001
global_hidden_units1 = 200
global_hidden_units2 = 200
global_use_dropout = False

# Declare placeholders to give them global scope
keep_prob = None
x = None
y_actuals = None

# Initializes variables within the TF graph before each experiment
def initialize_graph():
  global keep_prob, x, y_actuals

  # Placeholders
  keep_prob = tf.placeholder(tf.float32)
  x = tf.placeholder(tf.float32, [None, 784])
  y_actuals = tf.placeholder(tf.float32, [None, 10])

  weights = {
  'W1': tf.Variable(
    tf.truncated_normal(shape=[NUM_PIXELS, global_hidden_units1],
                        stddev=1.0/np.sqrt(NUM_PIXELS))),
  'W2': tf.Variable(
    tf.truncated_normal(shape=[global_hidden_units1, global_hidden_units2], 
                        stddev=1.0/np.sqrt(global_hidden_units1))),
  'W3': tf.Variable(
    tf.truncated_normal(shape=[global_hidden_units2, global_hidden_units3], 
                        stddev=1.0/np.sqrt(global_hidden_units2))),
  'W4': tf.Variable(
    tf.truncated_normal(shape=[global_hidden_units3, OUTPUT_UNITS], 
                        stddev=1.0/np.sqrt(global_hidden_units3)))
  }

  biases = {
    'b1': tf.Variable(tf.fill([global_hidden_units1], 0.01)),
    'b2': tf.Variable(tf.fill([global_hidden_units2], 0.01)),
    'b3': tf.Variable(tf.fill([global_hidden_units3], 0.01)),
    'b4': tf.Variable(tf.fill([OUTPUT_UNITS], 0.01))
  }

  # Construct model
  predictions = build_model(x, weights, biases, keep_prob)

  # Define cost function on predictions and optmizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y_actuals))
  optimizer = tf.train.AdamOptimizer(learning_rate=global_learning_rate).minimize(cost)
  return predictions, optimizer

# Neural network model
def build_model(x, weights, biases, keep_prob):
  
  # First hidden layer (with ReLU)
  z_1 = tf.matmul(x, weights['W1']) + biases['b1']
  h_1 = tf.nn.relu(z_1)
  h_1_drop = tf.nn.dropout(h_1, keep_prob)

  # Second hidden layer (with ReLU)
  z_2 = tf.matmul(h_1_drop, weights['W2']) + biases['b2']
  h_2 = tf.nn.relu(z_2)
  h_2_drop = tf.nn.dropout(h_2, keep_prob)

  # Third hidden layer (with ReLU)
  z_3 = tf.matmul(h_2_drop, weights['W3']) + biases['b3']
  h_3 = tf.nn.relu(z_3)
  h_3_drop = tf.nn.dropout(h_3, keep_prob)

  # Output layer, minus softmax
  z_4 = tf.matmul(h_3_drop, weights['W4']) + biases['b4']
  return z_4

# Runs Tensorflow session using predefined optimizer and hyperparameters
# Returns testing accuracy based on generated model
def train_and_evaluate():

  predictions, optimizer = initialize_graph()

  # Define accuracy measures
  correct_predictions = tf.equal(tf.argmax(predictions,1), tf.argmax(y_actuals,1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

  with tf.Session() as session:
    tf.global_variables_initializer().run()

    print("Starting Training")
    for epoch in range(NUM_EPOCHS):

      total_batches = int(mnist.train.num_examples/BATCH_SIZE)

      # Train on whole randomised dataset looping over batches
      for _ in range(total_batches):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        #session.run([optimizers['SGD']], feed_dict = {
        session.run(optimizer, feed_dict = {
          x: batch_xs, 
          y_actuals: batch_ys,
          keep_prob: 0.5 if global_use_dropout else 1
        })

      # Display current cost
      if (epoch % 10 == 0):
        _print ("Epoch: %3d, Current Validation Set Accuracy: %.4f" % 
          (epoch,
           session.run(accuracy, feed_dict = {
                    x: mnist.validation.images, 
                    y_actuals: mnist.validation.labels,
                    keep_prob: 1.0
           })))

    print("Training Completed!")

    if optimize_hyperparameters:
      # Evaluate on validation dataset when done.
      fin_accuracy = session.run(accuracy, feed_dict = {
          x: mnist.validation.images, 
          y_actuals: mnist.validation.labels,
          keep_prob: 1.0 })
      _print("Accuracy on validation set after training: %.4f\n" % fin_accuracy)

    else:
      # Evaluate on testing dataset when done.
      fin_accuracy = session.run(accuracy, feed_dict = {
          x: mnist.test.images, 
          y_actuals: mnist.test.labels,
          keep_prob: 1.0 })
      _print("Accuracy on test set after training: %.4f\n" % fin_accuracy)

  tf.reset_default_graph()
  return fin_accuracy

def optimize():

  global global_learning_rate, global_hidden_units1, global_hidden_units2, global_hidden_units3, global_use_dropout
  best_acc = best_lr = best_hu1 = best_hu2 = best_kr = 0

  # Parameters settings to go through
  learning_rates = [0.001]
  hidden_units1_opts = [200]
  hidden_units2_opts = [200]
  hidden_units3_opts = [200]
  do_dropout = [False]

  # Loops over sets of parameters
  for i in range(len(learning_rates)):

    global_learning_rate = learning_rates[i]
    global_hidden_units1 = hidden_units1_opts[i]
    global_hidden_units2 = hidden_units2_opts[i]
    global_hidden_units3 = hidden_units3_opts[i]
    global_use_dropout = do_dropout[i]

    _print("Training with:\nLR=%.3f, #HU1=%2d, #HU2=%2d, #HU3=%2d, Dropout=%r\n" % 
          (global_learning_rate, global_hidden_units1, global_hidden_units2, global_hidden_units3, global_use_dropout))

    test_acc, val_acc = train_and_evaluate()

    # If accuracy improved, store best results
    if val_acc > best_acc:
      best_acc = acc
      best_lr = global_learning_rate
      best_hu1 = global_hidden_units1
      best_hu2 = global_hidden_units2
      best_hu3 = global_hidden_units3
      best_momentum = global_momentum
      best_use_drop = global_use_dropout

  _print("Best Hyperparameter Values:\nLR=%.3f, #HU1=%2d, #HU2=%2d, #HU3=%2d, Dropout=%r" %
        (best_lr, best_hu1, best_hu2, best_hu3, best_use_drop))
  _print("Best Accuracy: %.4f\n" % best_acc)
  _print("Completed at %s" % str(datetime.datetime.now()).replace(' ', '_')[:19])
  logfile.close()


if __name__ == '__main__':
  if optimize_hyperparameters:
    optimize()
  
  else:
    train_and_evaluate()
