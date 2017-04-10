import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Fixed
NUM_ITERATIONS = 20001
BATCH_SIZE = 64

# Weight initialization
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
		                  strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# First convolutional layer
# The convolution will compute 32 features for each 5x5 path.
# Its weight tensor will have a shape of [5, 5, 1, 32].
# First two dim are patch size, next is number of input channels
# and the last is the number of output channels. We will also have a bias
# vector with a component for each output channel.

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Training
# Similar to simply one layer SoftMax but with differences:
# - Replace steepest gradient descent optimizer with more sophisticated ADAM optimizer.

cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize all variables
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Train model
for i in range(NUM_ITERATIONS):
	batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
	
	if i >= (NUM_ITERATIONS - 20) or i%20 == 0:
		train_accuracy = accuracy.eval(feed_dict={
			x: batch_xs, y_: batch_ys, keep_prob: 1.0})
		print("Iteration %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

# Evaluate model
print("Final test accuracy %g"%accuracy.eval(feed_dict={
	x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))




