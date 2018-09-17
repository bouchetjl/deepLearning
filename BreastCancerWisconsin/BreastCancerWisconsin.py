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

lesdatas = pd.read_csv('datasetBreastCancer.csv')
#print (lesdatas)

lesdatas['diagnosis'] = lesdatas['diagnosis'].map({'M':1,'B':0})
    
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if (feature_name not in ["id", "diagnosis"]):
            colonne=df[feature_name]
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

normalize_datas = normalize(lesdatas);

X_breastcancer=normalize_datas[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean",
                                "concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se",
                                "texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
                                "concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
                                "perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",
                                "concave points_worst","symmetry_worst","fractal_dimension_worst"]]

normalize_datas['NotDiagnosis'] = 1 - normalize_datas['diagnosis']

y_breastcancer=normalize_datas[["diagnosis", 'NotDiagnosis']]


X_train, X_test, y_train, y_test = train_test_split(X_breastcancer, y_breastcancer, test_size=0.2)


start_learning_rate=0.009
decay_steps=1000
decay_rate=1/2
global_step=tf.Variable (0, trainable=False,name="global_step")
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_rate)


# Parameters

training_epochs = 100
batch_size = 10
display_step = 3

# Neural Network Parameters

n_hidden_1 = 20
n_input = X_train.shape[1]
n_classes = y_train.shape[1]

# TensorFlow Graph input

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float")

# Create NN model

def neural_network(x, weights, biases):
    # Hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']

    #out_layer = tf.nn.dropout(out_layer, dropout)
    return out_layer

# Layers weight and bias
weights = {
    'h1': tf.Variable(tf.random_uniform(shape=(n_input, n_hidden_1),minval=-1, maxval=1, dtype=tf.float32, seed=0)),
    'out': tf.Variable(tf.random_uniform(shape=(n_hidden_1, n_classes),minval=-1, maxval=1, dtype=tf.float32, seed=0))
}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Constructing model

pred = neural_network(x, weights, biases)

# Defining loss and optimizer
#tf.summary.histogram('histogram_weights_couche1', weights['h1'])
#tf.summary.histogram('histogram_weights_couche2', weights['h2'])
#tf.summary.histogram('histogram_weights_couche3', weights['out'])

with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#tf.summary.scalar('cross_entropy', cross_entropy)

#tf.summary.scalar('learning_rate', learning_rate)
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy, global_step=global_step)

    #optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    #train=optimizer.minimize(cross_entropy, global_step=global_step)

    
# Initializing the variables

init = tf.global_variables_initializer()

# Running first session

with tf.Session() as sess:

    '''
    merged_summary = tf.summary.merge_all()
    # op to write logs to Tensorboard
    logs_path="test_logs"
    train_writer = tf.summary.FileWriter(logs_path + '/train',sess.graph)
    test_writer = tf.summary.FileWriter(logs_path + '/test')
    '''
    
    sess.run(init)
    # Training cycle
    avg_cost = 5.
    for epoch in range(training_epochs):
        
        total_batch = int(len(X_train) / batch_size)

        X_batches = np.array_split(X_train, total_batch)
        Y_batches = np.array_split(y_train, total_batch)

        if (epoch % display_step == 0) :
            print("epoch:", '%d' % (epoch + 1), "cost=", "{:.4f}".format(avg_cost))
            avg_cost=0.
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            str_accuracy=str(accuracy.eval({x: X_test, y: y_test}))
            
            '''
            test_summary = tf.Summary()
            test_summary.value.add(tag='Evaluation',simple_value=float(str_accuracy))
            test_writer.add_summary(test_summary, epoch)
            test_writer.flush()
            '''
            
            print("Accuracy:", str_accuracy)

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization operation (backprop) and cost operation(to get loss value)
            _, c, learnrate = sess.run([train, cross_entropy, learning_rate],
                                                feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            #train_writer.add_summary(summary, epoch*total_batch + i)

    #test_writer.close()
    #train_writer.close()
