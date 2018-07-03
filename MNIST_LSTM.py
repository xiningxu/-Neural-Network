# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 08:57:06 2018
Classify MNIST with LSTM
@author: XiningXu
"""

import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Import Dataset
DateSet_file_name = "MNIST_data"
mnist = input_data.read_data_sets(DateSet_file_name, one_hot=True)

# Parameter Setting
lr = 1e-4  # learning rate
iters = 50000 
batch_size = 128
n_input = 28 # length of input
time_step = 28 # length of a certain sequence
n_class = 10  #  output: number of classes
n_hidden = 128 # number of hidden units

x = tf.placeholder(dtype=tf.float32,shape=[None,time_step,n_input])
y = tf.placeholder(dtype=tf.float32,shape=[None,n_class])
# set parameters
weights = tf.Variable(tf.random_normal([n_hidden,n_class]),name='weights')
bias  = tf.Variable(tf.zeros([n_class]),name='bias')

def built_LSTM(x,weights,bias):
    # [batch_size,time_step,n_input]==>[batch_size*time_step,n_input]
    x = tf.unstack(x,time_step,axis=1)
    # build cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1)
    lstm_output , lstm_state = tf.contrib.rnn.static_rnn(lstm_cell,x,
                                                         dtype=tf.float32)
    output = tf.add(tf.matmul(lstm_output[-1],weights),bias)
    return output

y_pred = built_LSTM(x,weights,bias)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,
                                                              labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss) 
correct_pred = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    n_batch = int(mnist.train.num_examples/batch_size)
    test_data = mnist.test.images.reshape((-1,28,28))
    test_label = mnist.test.labels
    for i in range(iters):
        xs,ys = mnist.train.next_batch(batch_size)
        # xs shape: (None,784)=>(batch_size,time_step,n_input)
        xs = xs.reshape((batch_size,time_step,n_input))
        _,tmp_loss = sess.run([optimizer,loss],feed_dict={x:xs,y:ys})
        if i%100 == 0:
            train_batch_acc = sess.run(accuracy,feed_dict={x:xs,y:ys})
            print("Step: "+ str(i)+" Train Accuracy: "+ str(train_batch_acc))
        # test set 
        if i%500 == 0:
            test_loss,test_acc = sess.run([loss,accuracy],feed_dict={x:test_data,
                                  y:test_label})
            print("Step:" + str(i) + " Test Accuracy: "+ str(test_acc))            
    print("Optimization Finished !")
    test_loss,test_acc = sess.run([loss,accuracy],feed_dict={x:test_data,
                                  y:test_label})
    print('test_loss {:.3f},test_acc {:.3f}'.format(test_loss,test_acc))

    