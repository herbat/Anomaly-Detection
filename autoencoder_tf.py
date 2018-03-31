import tensorflow as tf
import numpy as np
import pickle
import math
import random

epochs = 100
datapoints = 1000
recordings = 5000
n_hidden = 2
batch_size = 128

eeg_data = pickle.load(open('trainingset.pkl', 'rb'))

def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def biases(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def fc_layer(prev, input_size, output_size):
    W = weights([input_size, output_size])
    b = biases([output_size])
    return tf.matmul(prev, W) + b

def autoencoder(x):
    l1 = tf.nn.tanh(fc_layer(x, 1000, 200))
    l2 = tf.nn.tanh(fc_layer(l1, 200, 200))
    l3 = fc_layer(l2, 200, 100)
    l4 = tf.nn.tanh(fc_layer(l3, 100, 200))
    l5 = tf.nn.tanh(fc_layer(l4, 200, 200))
    out = fc_layer(l5, 200, 1000)
    loss = tf.reduce_mean(tf.squared_difference(x, out))
    return loss, out


x = tf.placeholder(tf.float32, shape=[None, datapoints])

loss, output = autoencoder(x)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
print(loss)
print(train_step)
# run the training loop
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        batch = random.sample(eeg_data, batch_size)
        if i % 50 == 0:
            train_loss = sess.run(train_step, feed_dict={x: batch})
            print(train_loss)
            #print("Step: %d. Loss: %g" % (i, train_loss))

        thing = train_step.run(feed_dict={x: batch})

