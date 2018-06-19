import tensorflow as tf
import numpy as np
sess = tf.Session()

from keras import backend as K
K.set_session(sess)


img = tf.placeholder(tf.float32, shape=(None, 2), name='input_plhdr')


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', name='Intermediate'),
    tf.keras.layers.Dense(2, activation='softmax', name='Output'),
])

M = model(img)
print('input', img.name)
print('output', M.name)
sess.run(tf.global_variables_initializer())
print('result', sess.run(M, {img: np.array([[42, 43.]], dtype=np.float32)}))

saver = tf.train.Saver(tf.global_variables())
saver.save(sess, './exported/my_model')
tf.train.write_graph(sess.graph, '.', "./exported/graph.pb", as_text=False)