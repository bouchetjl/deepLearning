import csv
import tensorflow as tf
import random
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split


dataset_diabetes = pd.read_csv('diabetes.csv')

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if (feature_name not in ["PatientID", "Diabetic"]):
            colonne=df[feature_name]
            moyenne= df[feature_name].mean()
            ecarttype= df[feature_name].std()
            result[feature_name] = (df[feature_name] - moyenne) / ecarttype
    return result

def traite (df) :
    result = df.copy()
    for feature_name in df.columns:
        if (feature_name in ["Age"]):
            result[feature_name] = np.log(df[feature_name])
    return result
  
dataset_diabetes_traite = traite (dataset_diabetes)
dataset_diabetes_norm = normalize(dataset_diabetes_traite)

X_diabetes = dataset_diabetes_norm[
    ["Pregnancies","PlasmaGlucose","DiastolicBloodPressure","TricepsThickness","SerumInsulin","BMI","DiabetesPedigree","Age"]]


dataset_diabetes_norm['NotDiabetic'] = 1 - dataset_diabetes_norm['Diabetic']
y_diabetes = dataset_diabetes_norm[['Diabetic', 'NotDiabetic']]


X_train, X_test, y_train, y_test = train_test_split(X_diabetes, y_diabetes, test_size=0.2,random_state=123)




# Parameters
training_epochs = 300
batch_size = 20
display_step = 5

# Neural Network Parameters
n_hidden_1 = 20
n_hidden_2 = 20
n_input = X_train.shape[1]
n_classes = y_train.shape[1]
dropout = 0.01

#learning rate
start_learning_rate=0.01
decay_steps=100000
decay_rate=1/12
global_step=tf.Variable (0, trainable=False,name="global_step")
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_rate)


# TensorFlow Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float")

# Create NN model
def neural_network(x, weights, biases):
    
    # Hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    #out_layer = tf.nn.dropout(out_layer, dropout)
    return out_layer

# Layers weight and bias
weights = {
    'w1': tf.Variable(tf.random_uniform(shape=(n_input, n_hidden_1),minval=-1, maxval=1, dtype=tf.float32, seed=0)),
    'w2': tf.Variable(tf.random_uniform(shape=(n_hidden_1, n_hidden_2),minval=-1, maxval=1, dtype=tf.float32, seed=0)),
    'out': tf.Variable(tf.random_uniform(shape=(n_hidden_2, n_classes),minval=-1, maxval=1, dtype=tf.float32, seed=0))
}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# compute predictions from model
pred = neural_network(x, weights, biases)

# put weights in summary
tf.summary.histogram('histogram_weights_couche1', weights['w1'])
tf.summary.histogram('histogram_weights_couche2', weights['w2'])
tf.summary.histogram('histogram_weights_couche3', weights['out'])

# compute cross entropy
with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#tf.summary.scalar('cross_entropy', cross_entropy)

# compute training
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy, global_step=global_step)

    
# Initializing the variables
init = tf.global_variables_initializer()

# Running first session
with tf.Session() as sess:

    # summaries for tensorboard
    merged_summary = tf.summary.merge_all()
    logs_path="test_logs"
    train_writer = tf.summary.FileWriter(logs_path + '/train',sess.graph)
    test_writer = tf.summary.FileWriter(logs_path + '/test')

    sess.run(init)
 
    # first average cost for console
    first_cost, learnrate = sess.run([cross_entropy, learning_rate], feed_dict={x: X_test, y: y_test})
    print("epoch: 1 cost=", "{:.4f}".format(first_cost))
    
    # average cost per epoch group for console
    cpt_avg_cost = 0
    somme_avg_cost = 0.
    
    # average cross entropy during a period for tensorboard
    somme_cross_entropy = 0.
    cpt_cross_entropy = 0

    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(len(X_train) / batch_size)

        X_batches = np.array_split(X_train, total_batch)
        Y_batches = np.array_split(y_train, total_batch)

        # Display logs per epoch step
        if (epoch % display_step == 0) :
            
            if (epoch!=0):
                avg_cost = somme_avg_cost/cpt_avg_cost
                print("epoch:", '%d' % (epoch + 1), "cost=", "{:.4f}".format(avg_cost))
                somme_avg_cost=0.
                cpt_avg_cost=0
                                        
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            str_accuracy=str(accuracy.eval({x: X_test, y: y_test}))
            
            # put evaluation un test summary
            test_summary = tf.Summary()
            test_summary.value.add(tag='Evaluation',simple_value=float(str_accuracy))
            test_writer.add_summary(test_summary, epoch)
            test_writer.flush()
            
            # accuracy on console
            print ("learning_rate : " + str(learnrate))
            print("Accuracy:", str_accuracy)
                        
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization operation (backprop) and cost operation(to get loss value)
            _, c, learnrate, summary = sess.run([optimizer, cross_entropy, learning_rate, merged_summary],
                                                feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            somme_avg_cost += c
            cpt_avg_cost+=1
            
            somme_cross_entropy +=c
            cpt_cross_entropy+=1
            
            # average cross entropy in train summary
            if (cpt_cross_entropy%20 == 0) :
                
                moyenne_cross_entropy=somme_cross_entropy/cpt_cross_entropy
                train_summary = tf.Summary()
                train_summary.value.add(tag='Cross entropy',simple_value=moyenne_cross_entropy)
                train_summary.value.add(tag='Learning rate',simple_value=learnrate)
                somme_cross_entropy = 0.
                cpt_cross_entropy = 0
                
                train_writer.add_summary(train_summary, epoch*total_batch + i)
            
            train_writer.add_summary(summary, epoch*total_batch + i)


    # end cleaning
    test_writer.close()
    train_writer.close()
