# using tis tf.compat thing to run session in tensorflow 2.0, because normal session running is depcriciated in this one


import tensorflow as tf
tf.compat.v1.disable_eager_execution()


# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0)

# # tf.print([node1,node2])
# #performing mathematical operations here


# # a = tf.constant(5)
# # b = tf.constant(2)
# # c = tf.constant(3)

# # d = tf.multiply(a,b)
# # e =tf.add(c,b)
# # f = tf.subtract(d,e)

# # sess = tf.compat.v1.Session()
# # outs = sess.run(f)
# # print(outs)


# #now not using constant value, now we are using the placeholders

# # a = tf.compat.v1.placeholder(tf.float32)
# # b = tf.compat.v1.placeholder(tf.float32)

# # adder_node = a+b

# # sess = tf.compat.v1.Session()

# # print(sess.run(adder_node, {a:[1,3], b:[2,4]}))


# #now using bias, wieghts i.e variables together with placeholders

# W = tf.Variable([.3], tf.float32)
# b = tf.Variable([-.3], tf.float32)

# x = tf.compat.v1.placeholder(tf.float32)

# linear_model  = W*x + b

# init = tf.compat.v1.global_variables_initializer()

# sess = tf.compat.v1.Session()

# #sess.run(init)

# #print(sess.run(linear_model, {x:[1,2,3,4]}))

# #now checking the efficiency of our above model/function by calculating the loss

# y = tf.compat.v1.placeholder(tf.float32)

# sqaured_deltas = tf.square(linear_model-y)

# loss = tf.reduce_sum(sqaured_deltas)

# #print(sess.run(loss, {x:[1,2,3,4] , y:[0,-1,-2,-3]}))

# # now we are going to optimise this loss

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01)

# train = optimizer.minimize(loss)

# sess.run(init)

# for i in range(1000):
#     sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

# print(sess.run([W,b]))



