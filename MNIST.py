import tensorflow as tf
# import tensorflow_datasets as tfds
#from tensorflow.examples.tutorials.mnist import input_data
# mnist = tfds.load(name="mnist", split=tfds.Split.TRAIN)
mnist = tf.keras.datasets.mnist

sess = tf.compat.v1.InteractiveSession()

x = tf.compat.v1.placeholder(tf.float32, shape=[None,784])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.compat.v1.global_variables_initializer())

y_ = tf.matmul(x,w) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
)

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_bacth(100)
    train_step.run(feed_dict={x:batch[0], y:batch[1]})

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels}))