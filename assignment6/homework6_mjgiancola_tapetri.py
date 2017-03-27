import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()  


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

# initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# Creates all layers and interactions
def network(hidden_units1, hidden_units2, keep_prob = 0.5):

  x = tf.placeholder(tf.float32, shape=[None, 784])

  # Hidden layer 1
  W_1 = tf.Variable()
  b_1 = tf.Variable()
  z_1 = tf.matmul(x, W_1) + b_1
  h_1 = tf.nn.relu(z_1)
  h_1_drop = tf.nn.dropout(h_1, keep_prob)

  # Hidden layer 2
  W_2 = tf.Variable()
  b_2 = tf.Variable()
  z_2 = tf.matmul(h_1_drop, W_2) + b_2
  h_2 = tf.nn.relu(z_2)
  h_2_drop = tf.nn.dropout(h_2, keep_prob)

  # Output layer
  W_3 = tf.Variable()
  b_3 = tf.Variable()
  z_3 = tf.matmul(h_2_drop, W_3) + b_3

  y_hats = z_3 # something something numerical stability

  return y_hats





# Creates the training operation 
def train(optimizer, y_hats, num_epochs=1000):

  # Setup variables used for training
  tf.global_variables_initializer().run()

  # Used for evaluation during training
  y_actuals = tf.placeholder(tf.float32, [None, 10])

  # Loss function is cross entropy
  cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_actuals, logits=y_hats) )
  train_step = optimizer.minimize(cross_entropy)

  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_actuals: batch_ys})



# Return accuracy measure
def evaluation():


if __name__ == '__main__':


  x = tf.placeholder(tf.float32, [None, 784]) # Here, None means the dimension can be of any length

  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))

  y_hats = tf.matmul(x, W) + b
  y_actuals = tf.placeholder(tf.float32, [None, 10])


  correct_prediction = tf.equal(tf.argmax(y_hats,1), tf.argmax(y_actuals,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_actuals: mnist.test.labels}))


"""
learning rate schedule, momentum, minibatch size, optimizer
"""
