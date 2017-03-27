import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':

  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  x = tf.placeholder(tf.float32, [None, 784]) # Here, None means the dimension can be of any length
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))

  y_hats = tf.matmul(x, W) + b
  y_actuals = tf.placeholder(tf.float32, [None, 10])

  # First is numerically unstable
  # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actuals * tf.log(y_hats), reduction_indices=[1]))
  cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_actuals, logits=y_hats) )

  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_actuals: batch_ys})

  correct_prediction = tf.equal(tf.argmax(y_hats,1), tf.argmax(y_actuals,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_actuals: mnist.test.labels}))
