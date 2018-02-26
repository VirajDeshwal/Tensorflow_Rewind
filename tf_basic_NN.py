#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:02:10 2018

@author: virajdeshwal
"""

''' 
Let's create a simple Neural Network in Tensorflow'''
import tensorflow as tf
import numpy as np

n_features = 10
n_dense_neurons =3

x = tf.placeholder(tf.float32, (None, n_features))

w = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
b = tf.Variable(tf.random_normal([n_dense_neurons]))


'''Now let's bind the operations together'''
wx = tf.matmul(x,w)
y = tf.add(wx, b)

a = tf.sigmoid(y)
#initializing the global session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    layer_out = sess.run(a, feed_dict={x:np.random.random([1,n_features])})
print(layer_out)

'''The above model is working for fixed values of w and b.
Now let's adjust the w and b.'''

#Ler's add some noise to the X_data
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_labels = np.linspace(0,10,10)+np.random.uniform(-1.5,1.5,10)


''' 
Now lets visualise the labels(y) and data(x)
'''
import matplotlib.pyplot as plt
plt.xlabel('data points')
plt.ylabel('labels')
plt.title('Plotting the random points')
plt.plot(x_data, y_labels,'*')

'''Plotting is done Now lets fit the curve for
            y=mx+b'''

''' Generating the two random variables for m and b '''
two_random_numbers=np.random.rand(2)
print('The two random variables generated will be used as weight and bias are {}'.format(two_random_numbers))

m = tf.Variable(0.75)
b = tf.Variable(.24)
''' for now let's put the error to zero'''
error =0
for x,y in zip(x_data, y_labels):
    y_hat = m*x+b
    error+= (y-y_hat)**2
''' Now let's minimize the error by optimizing the 
model'''
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = tf.global_variables_initializer()


'''Let's create the global session to run the model'''
with tf.Session() as sess:
    sess.run(init)
    training_steps =1000
    for i in range (training_steps):
        sess.run(train)
        final_slope, final_intercept = sess.run([m,b])

x_test = np.linspace(-1,11,10)
#y = mx+b
y_pred_plot = final_slope*x_test+final_intercept
plt.plot(x_test, y_labels, color='red')
plt.plot(x_data,y_labels,'*')
    

''' Let's use the estimator API'''
feat_col = [tf.feature_column.numeric_column('x', shape = [1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_col)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y,
                                                    test_size =0.2)
print(x_train.shape)

''' We need an input function which can act like a dict which can take feed fucntion and batch size at once'''
input_func = tf.estimator.inputs.numpy_io({'x':x_train}, y_train,
                                          batch_size=8,
                                          num_epochs=None, shuffle=True)
