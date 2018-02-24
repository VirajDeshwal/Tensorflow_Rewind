#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:52:27 2018

@author: virajdeshwal
"""

import tensorflow as tf
import numpy as np

'''
Let's create some random variables to feed into the placeholders'''

rand_a = np.random.uniform(0.0, 100.0, size = (5,5))
rand_b = np.random.uniform(0.0, 100.0, size = (5,1))

#Placeholders are ready.

'''Let's create some Placeholders to feed the values 
    from variables'''

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

'''
Let's create some operation to perform on these placeholders
'''

add = a+b
mul = a*b

'''
Let's create the session
'''

with tf.Session() as sess:
    results = sess.run(add, feed_dict={a:rand_a, b:rand_b})
    print('The addition of the two matrics:',results)

with tf.Session() as sess:
    result=sess.run(mul,feed_dict={a:rand_a, b:rand_b})
    print('The multiplication of two matrics:',result)
    
    
    