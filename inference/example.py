import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=[1, 2], name='input')
output = tf.identity(tf.layers.dense(x, 1), name='output')

val = np.array([[1, 1]], dtype=np.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # save graph
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, './exported/my_model')

    tf.train.write_graph(sess.graph, '.', "./exported/graph.pb", as_text=False)
    tf.train.write_graph(sess.graph, '.', "./exported/graph.pb_txt", as_text=True)

    t1 = tf.get_default_graph().get_tensor_by_name('output:0')
    t2 = tf.get_default_graph().get_tensor_by_name('dense/kernel:0')
    t3 = tf.get_default_graph().get_tensor_by_name('dense/bias:0')

    t1, t2, t3, x = sess.run([t1, t2, t3, x], {x: val})

    print(tf.global_variables())
    print("input           ", x)
    print("output          ", t1)
    print("dense/kernel:0  ", t2)
    print("dense/bias:0    ", t3)
