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

"""
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

# Global Parameters to optimize
global_learning_rate = 0.5
global_hidden_units1 = 50
global_hidden_units2 = 20
global_momentum = 0.1
global_use_dropout = True

keep_prob = None
x = None
y_actuals = None

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
    tf.truncated_normal(shape=[global_hidden_units2, OUTPUT_UNITS], 
                        stddev=1.0/np.sqrt(global_hidden_units2)))
  }

  biases = {
    'b1': tf.Variable(tf.fill([global_hidden_units1], 0.1)),
    'b2': tf.Variable(tf.fill([global_hidden_units2], 0.1)),
    'b3': tf.Variable(tf.fill([OUTPUT_UNITS], 0.1))
  }

  # Construct model
  predictions = build_model(x, weights, biases, keep_prob)

  # Define cost function on predictions and optmizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y_actuals))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=global_learning_rate).minimize(cost)
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

  # Output layer, minus softmax
  z_3 = tf.matmul(h_2_drop, weights['W3']) + biases['b3']
  return z_3

# optimizers = {
#   'SGD' : tf.train.GradientDescentOptimizer(global_learning_rate=global_learning_rate).minimize(cost),
#   'Momentum' : tf.train.MomentumOptimizer(global_learning_rate=global_learning_rate, momentum=global_momentum).minimize(cost),
#   # 'Adam' : tf.train.GradientDescentOptimizer(global_learning_rate=global_learning_rate).minimize(cost)

# }


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
        _print ("Epoch: %3d, Current Training Set Accuracy: %.4f" % 
          (epoch,
           session.run(accuracy, feed_dict = {
                    x: mnist.train.images, 
                    y_actuals: mnist.train.labels,
                    keep_prob: 1.0
           })))

    print("Training Completed!")

    # Evaluate on test dataset when done.
    test_accuracy = session.run(accuracy, feed_dict = {
        x: mnist.test.images, 
        y_actuals: mnist.test.labels,
        keep_prob: 1.0 })
    _print("Accuracy on test set after training: %.4f\n" % test_accuracy)

  tf.reset_default_graph()
  return test_accuracy

def optimize():

  global global_learning_rate, global_hidden_units1, global_hidden_units2, global_momentum, global_use_dropout
  best_acc = best_lr = best_hu1 = best_hu2 = best_momentum = best_kr = 0

  # Parameters to go through
  learning_rates = [0.01, 0.1, 0.5]
  hidden_units1_opts = [60, 40, 20]
  hidden_units2_opts = [30, 20, 10]
  momentum_opts = [0.001, 0.01, 0.1, 0.5]
  do_dropout = [True, False]

  for current_lr in learning_rates:
    for current_hidden_1 in hidden_units1_opts:
      for current_hidden_2 in hidden_units2_opts:
        for drop in do_dropout:

          global_learning_rate = current_lr
          global_hidden_units1 = current_hidden_1
          global_hidden_units2 = current_hidden_2
          #global_momentum = current_momentum
          global_use_dropout = drop

          _print("Training with:\nLR=%.3f, #HU1=%2d, #HU2=%2d, Dropout?=%r\n" % 
                (global_learning_rate, global_hidden_units1, global_hidden_units2, global_use_dropout))

          # _print("Training with:\nLR=%.3f, #HU1=%2d, #HU2=%2d, Momentum=%.3f, Dropout?=%r\n" % 
          #       (global_learning_rate, global_hidden_units1, global_hidden_units2, global_momentum, global_use_dropout))

          acc = train_and_evaluate()

          # If accuracy improved, store best results
          if acc > best_acc:
            best_acc = acc
            best_lr = global_learning_rate
            best_hu1 = global_hidden_units1
            best_hu2 = global_hidden_units2
            best_momentum = global_momentum
            best_use_drop = global_use_dropout

  _print("Best Hyperparameter Values:\nLR=%.3f, #HU1=%2d, #HU2=%2d, Dropout=%r" %
        (best_lr, best_hu1, best_hu2, best_use_drop))
  # _print("Best Hyperparameter Values:\nLR=%.3f, #HU1=%2d, #HU2=%2d, Momentum=%.3f, Dropout=%r" %
  #       (best_lr, best_hu1, best_hu2, best_momentum, best_use_drop))
  _print("Completed at %s" % str(datetime.datetime.now()).replace(' ', '_')[:19])
  logfile.close()


if __name__ == '__main__':
  if 1: # Set to 1 to optimize hyperparameters
    optimize()
  
  else: # If above set to 0, just run on default global values
    train_and_evaluate()
