import tensorflow as tf

x = tf.placeholder(tf.float32)
w = tf.get_variable('w', initializer=1.)
y = tf.add(x, w, name='y')

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run(y, {x: 5}))

  builder = tf.saved_model.builder.SavedModelBuilder('/tmp/simple_model/1')

  tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
  tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'input': tensor_info_x},
              outputs={'output': tensor_info_y},
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
      },
      legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op'))

  builder.save()
