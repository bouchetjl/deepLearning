# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

from subprocess import check_output

import csv
import tensorflow as tf
import random
import pandas as pd
import numpy as np
from time import time

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

from tensorflow.contrib import layers
from tensorflow.contrib import learn

lesnumerations = pd.read_csv('datasetMumeration.csv')

X_numerations = lesnumerations[
    ['Hematies', 'Leucocytes', 'Hemoglobine', 'Sexe']]

y_numerations = lesnumerations[['Alerte']]

lesnumerations['NotAlerte'] = 1 - lesnumerations['Alerte']
y_numerations = lesnumerations[['Alerte', 'NotAlerte']]


X_train, X_test, y_train, y_test = train_test_split(X_numerations, y_numerations, test_size=0.1,
                                                    random_state=0)


start_learning_rate=0.05
decay_steps=50000
decay_rate=1/20
global_step=tf.Variable (0, trainable=False,name="global_step")
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_rate)


# Parameters

training_epochs = 650
batch_size = 25
display_step = 10

# Neural Network Parameters

n_hidden_1 = 8
n_hidden_2 = 8
n_input = X_train.shape[1]
n_classes = y_train.shape[1]
dropout = 0.01

# TensorFlow Graph input

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float")

# Create NN model

def neural_network(x, weights, biases):
    # Hidden layer with relu activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with relu activation
    
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    #out_layer = tf.nn.dropout(out_layer, dropout)
    return out_layer

# Layers weight and bias

weights = {
#    'h1': tf.Variable(tf.random_uniform(shape=(n_input, n_hidden_1),minval=-5, maxval=5, dtype=tf.float32, seed=0)),
    'h1': tf.Variable(tf.random_uniform(shape=(n_input, n_hidden_1),minval=-5, maxval=5, dtype=tf.float32, seed=0)),
    'h2': tf.Variable(tf.random_uniform(shape=(n_hidden_1, n_hidden_2),minval=-5, maxval=5, dtype=tf.float32, seed=0)),
    'out': tf.Variable(tf.random_uniform(shape=(n_hidden_2, n_classes),minval=-5, maxval=5, dtype=tf.float32, seed=0))
}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Constructing model

pred = neural_network(x, weights, biases)

# Defining loss and optimizer
tf.summary.histogram('histogram_weights_couche1', weights['h1'])
tf.summary.histogram('histogram_weights_couche2', weights['h2'])
tf.summary.histogram('histogram_weights_couche3', weights['out'])

with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
tf.summary.scalar('cross_entropy', cross_entropy)

tf.summary.scalar('learning_rate', learning_rate)
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy, global_step=global_step)

    
# Initializing the variables

init = tf.global_variables_initializer()

# Running first session

with tf.Session() as sess:

    merged_summary = tf.summary.merge_all()
    # op to write logs to Tensorboard
    logs_path="test_logs"
    train_writer = tf.summary.FileWriter(logs_path + '/train',sess.graph)
    test_writer = tf.summary.FileWriter(logs_path + '/test')

    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train) / batch_size)

        X_batches = np.array_split(X_train, total_batch)
        Y_batches = np.array_split(y_train, total_batch)

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization operation (backprop) and cost operation(to get loss value)
            _, c, learnrate, summary = sess.run([optimizer, cross_entropy, learning_rate, merged_summary],
                                                feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            train_writer.add_summary(summary, epoch*total_batch + i)
        # Display logs per epoch step
        if (epoch==1) or (epoch % display_step == 0) :
            print("epoch:", '%d' % (epoch + 1), "cost=", "{:.4f}".format(avg_cost))
            
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            str_accuracy=str(accuracy.eval({x: X_test, y: y_test}))
            
            test_summary = tf.Summary()
            test_summary.value.add(tag='Evaluation',simple_value=float(str_accuracy))
            test_writer.add_summary(test_summary, epoch)
            test_writer.flush()
            
            print("Accuracy:", str_accuracy)
            
    test_writer.close()
    train_writer.close()