#!/usr/bin/env python
# ComputerGraphics Tuebingen, 2018

import numpy as np
import tensorflow as tf
from user_ops import matrix_add

np.random.seed(42)
tf.set_random_seed(42)

matA = np.random.randn(1, 2, 3, 4).astype(np.float32) * 10
matB = np.random.randn(1, 2, 3, 4).astype(np.float32) * 10


A = tf.placeholder(tf.float32, shape=[None, 2, 3, 4])
B = tf.placeholder(tf.float32, shape=[None, 2, 3, 4])

bias = 42.

actual_op = matrix_add(A, B, bias)


with tf.Session() as sess:
    print sess.run(actual_op, {A: matA, B: matB})
