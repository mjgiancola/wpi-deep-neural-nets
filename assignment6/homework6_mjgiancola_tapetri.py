import tensorflow as tf

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

# Parameters 
learning_rate = 0.05


# Placeholders
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

# Can add various initialization procedures here
weights = {
  'W1': tf.Variable()
  'W2': tf.Variable()
  'W3': tf.Variable()
}

biases = {
  'b1': tf.Variable()
  'b2': tf.Variable()
  'b3': tf.Variable()
}

# Construct model
predictions = NN(x, weights, biases)

# Define cost function on predictions and optmizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as session:
  tf.initialize_all_variables().run()

  print("Starting Training")
  for epoch in range(num_epochs):

    total_batches = int(mnist.train.num_examples/batch_size)

    # Train on whole randomised dataset looping over batches
    for _ in range(total_batches)
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      session.run([train_step], feed_dict = {
        x: batch_xs, 
        y_actuals: batch_ys,
        keep_prob: 0.5
      })

    # Display current cost
    if (epoch % 100 == 0):
      print ("Loss")

  print("Training Completed!")

  # Evaluate on test dataset when done.
  correct_predictionss = tf.equal(tf.argmax(predictions,1), tf.argmax(y_actuals,1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
  print(sess.run(accuracy, feed_dict = {
    x: mnist.test.images, 
    y_actuals: mnist.test.labels
    keep_prob: 1.0
  }))

"""
learning rate schedule, momentum, minibatch size, optimizer
"""
