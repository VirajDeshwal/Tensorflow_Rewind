#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:02:56 2018

@author: virajdeshwal
"""

import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)

with tf.Session() as sess:
    result = sess.run(a+b)
print(result)


#Let.s print the default graph
print(tf.get_default_graph)

''' Now lets create our default graphs'''

g = tf.Graph()
print('the manual graph created by us :', g)


make_default = tf.get_default_graph()
print('The default graph create by us :',make_default)

'''
Now let's try to recall the default graph from Tensor and check whether 
it is called the default graph created by us or not.'''
print(tf.get_default_graph())
