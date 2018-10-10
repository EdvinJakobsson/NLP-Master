import tensorflow as tf

W = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)
c = tf.Variable([1.0],tf.float32)
learning_rate = 0.01

x = tf.placeholder(tf.float32)

linear_model = W * x + b - c

y = tf.placeholder(tf.float32)

# loss
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

# optimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
    print(sess.run([W,b,c]))