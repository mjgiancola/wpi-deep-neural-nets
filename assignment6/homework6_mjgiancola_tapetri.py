import tensorflow as tf
import numpy as np


# Import MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


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
NUM_EPOCHS = 1
BATCH_SIZE = 128
NUM_PIXELS = 784
OUTPUT_UNITS = 10

# Global Parameters to optimize
learning_rate = 0.5
hidden_units1 = 50
hidden_units2 = 20



# Placeholders
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 784])
y_actuals = tf.placeholder(tf.float32, [None, 10])

# Neural network model
def NN(x, weights, biases, keep_prob):
  
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

# Can add various initialization procedures here
weights = {
  'W1': tf.Variable(
    tf.truncated_normal(shape=[NUM_PIXELS, hidden_units1], 
                        stddev=1.0/np.sqrt(NUM_PIXELS))),
  'W2': tf.Variable(
    tf.truncated_normal(shape=[hidden_units1, hidden_units2], 
                        stddev=1.0/np.sqrt(hidden_units1))),
  'W3': tf.Variable(
    tf.truncated_normal(shape=[hidden_units2, OUTPUT_UNITS], 
                        stddev=1.0/np.sqrt(hidden_units2)))
}

biases = {
  'b1': tf.Variable(tf.fill([hidden_units1], 0.1)),
  'b2': tf.Variable(tf.fill([hidden_units2], 0.1)),
  'b3': tf.Variable(tf.fill([OUTPUT_UNITS], 0.1))
}

# Construct model
predictions = NN(x, weights, biases, keep_prob)

# Define cost function on predictions and optmizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y_actuals))
optimizers = {
  'SGD' : tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost),
  # 'Momentum' : tf.train.MomentumOptimizer(learning_rate=learning_rate).minimize(cost),
  # 'Adam' : tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

}


# Define accuracy measures
correct_predictions = tf.equal(tf.argmax(predictions,1), tf.argmax(y_actuals,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train_and_evaluate():

  with tf.Session() as session:
    tf.initialize_all_variables().run()

    print("Starting Training")
    for epoch in range(NUM_EPOCHS):

      total_batches = int(mnist.train.num_examples/BATCH_SIZE)

      # Train on whole randomised dataset looping over batches
      for _ in range(total_batches):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        session.run([optimizers['SGD']], feed_dict = {
          x: batch_xs, 
          y_actuals: batch_ys,
          keep_prob: 0.5
        })

      # Display current cost
      if (epoch % 10 == 0):
        print ("Epoch: %3d, Current Training Set Accuracy: %.4f" % 
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
    print("Accuracy on test set after training: %.4f" % test_accuracy)

    return test_accuracy

def main():

  # Parameters to go through
  learning_rates = [0.001, 0.01, 0.1, np.inf]
  # drop out or nah

  for current_lr in  learning_rates:
    learning_rate = current_lr
    train_and_evaluate()





if __name__ == '__main__':
  main()




