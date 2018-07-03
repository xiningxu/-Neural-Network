# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 

CNN for MNIST

with 2 convolutionary layers and a fc layer
node for 1st Layer: 32
node for 2nd Layer: 64
node for FC Layer: 512

@author: XiningXu
"""
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def init_weight_variable(shape):
    # Fun: initialize weights
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def init_bias_variable(shape):
    #  Fun: initialize bias
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')

# import dataset
DateSet_file_name = "MNIST_data"
mnist = input_data.read_data_sets(DateSet_file_name, one_hot=True)

# Parameter Setting
    # Training Parameters
batch_size = 50
test_size =  1000
learning_rate = 1e-4
    # Network Parameters
n_size = 28
n_class = 10

n_kernel = 5
n_channel_0 = 1
n_channel_1 = 32
n_channel_2 = 64
n_node = 512
n_input = 7*7*n_channel_2

# Establish Model
x = tf.placeholder(tf.float32,shape = [None,n_size*n_size])
y_hat = tf.placeholder(tf.float32,shape = [None,n_class])
x_input = tf.reshape(x, [-1,n_size,n_size,n_channel_0])#-1 denotes orignal size

# 1st Layer: 1 conv layer + 1 pooling layer
    # kernel size: [height,width,channel0,channel1]
w_1 = init_weight_variable([n_kernel,n_kernel,n_channel_0,n_channel_1])
b_1 = init_bias_variable([n_channel_1])
h_conv1 = tf.nn.relu(conv2d(x_input, w_1) + b_1)
h_pool1 = max_pool_2x2(h_conv1)

# 2nd Layer: 1 conv layer + 1 pooling layer
    # kernel size: [height,width,channel1,channel2]
w_2 = init_weight_variable([n_kernel,n_kernel,n_channel_1,n_channel_2])
b_2 = init_bias_variable([n_channel_2])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_2)+b_2)
h_pool2 = max_pool_2x2(h_conv2)

# 3rd Layer: FC Layer
w_fc1 = init_weight_variable([n_input,n_node])
b_fc1 = init_bias_variable([n_node])
    # Input Layer: mat2vec
h_pool2_flat = tf.reshape(h_pool2,[-1,n_input])
    # Hiddenlayer
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1) 
    # add drop to FC layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = init_weight_variable([n_node, n_class])
b_fc2 = init_bias_variable([n_class])

y_output = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# Cost Function
cross_entropy = -tf.reduce_sum( y_hat*tf.log(y_output) )
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal( tf.argmax(y_output,1),tf.argmax(y_hat,1) )
accuracy = tf.reduce_mean( tf.cast(correct_prediction,tf.float32) )

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Launch the graph in a session
for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    if i%100 == 0:
        #train_accuracy = accuracy.eval(feed_dict = {x:batch_xs,y_hat:batch_ys,
        #                                            keep_prob:1.0})
        #print ("step %d,training accuracy %.5f"%(i,train_accuracy))
        print("step %d"%i)
    train_step.run(feed_dict={x:batch_xs,y_hat:batch_ys,keep_prob:0.5})

for i in range(10):    
    test_batct_xs,test_batch_ys = mnist.test.next_batch(1000)    
    print("test accuracy %g"%accuracy.eval(feed_dict={ x:  test_batct_xs, 
                                                      y_hat: test_batch_ys, 
                                                      keep_prob: 1.0}))    
print("1")
sess.close()

