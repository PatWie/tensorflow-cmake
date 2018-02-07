import tensorflow as tf
import numpy as np

val = np.array([[1, 1]], dtype=np.float32)

with tf.Session() as sess:

    # load the computation graph
    loader = tf.train.import_meta_graph('./exported/my_model.meta')
    sess.run(tf.global_variables_initializer())
    loader = loader.restore(sess, './exported/my_model')

    x = tf.get_default_graph().get_tensor_by_name('input:0')
    t1 = tf.get_default_graph().get_tensor_by_name('output:0')
    t2 = tf.get_default_graph().get_tensor_by_name('dense/kernel:0')
    t3 = tf.get_default_graph().get_tensor_by_name('dense/bias:0')

    t1, t2, t3, x = sess.run([t1, t2, t3, x], {x: val})

    print tf.global_variables()
    print "input           ", x
    print "output          ", t1
    print "dense/kernel:0  ", t2
    print "dense/bias:0    ", t3
