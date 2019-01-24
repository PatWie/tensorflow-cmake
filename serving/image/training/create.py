import tensorflow as tf
import tensorflow.saved_model as sm


def pre_process_input(x):
  """Prepare received inputs.

  Remarks:
    TensorFlow-Serving expects encoded image data, but the model
    itself relies on decoded data. This function maps between both.

  Args:
      x (tf.tensor): A received unprocessed tensor.

  Returns:
      tf.tensor: A tensor, which is expected by the network itself.
  """

  # this is ugly, but fixe a few headaches
  x = tf.reshape(x, [])
  # decode the buffer as an image
  x = tf.image.decode_image(x, channels=3, dtype=tf.float32)
  # add the batch dimension and scale to [0, 1]
  x = tf.expand_dims(x, axis=0) / 255.
  return x


def post_process_output(y):
  """Prepare outputs before sending them back.

  Remarks:
    The model might produce an image, but we need to deliver the image encoded,
    such that a REST user can handle the response.

  Todo:
    In contrast to the official docs, the outputs has *not* been encoded to
    base64 similar to the input. Therefore, we explicitly encode it here.

  Args:
      y (tf.tensor): A tensor, which is produced by our model.

  Returns:
      tf.tensor: A tensor, which is ready to be delivered as a response.
  """

  # remove batch dimension
  y = tf.squeeze(y, axis=[0])
  # clip to correct range
  y = tf.clip_by_value(y, 0, 1) * 255.
  # Convert image to dtype, scaling its values if needed (is needed!).
  y = tf.image.convert_image_dtype(y, tf.uint8)
  # output is png file
  y = tf.image.encode_png(y)
  # add batch-dimension back
  y = tf.expand_dims(y, axis=0)

  # NOTE: While the doc clearly states the output format of TF-Serving will
  # match the input format, this is not the case. Hence, we explicitly encode
  # the output. (`encode_base64` is like `urlsafe_b64encode`)
  y = tf.io.encode_base64(y, pad=True)
  return y


def model(x):
  # so far only the identity
  return x


# the placeholder has to have the postfix "_bytes".
input_bytes = tf.placeholder(tf.string, shape=[], name="input_bytes")
input_img = pre_process_input(input_bytes)
prediction = model(input_img)
output = post_process_output(prediction)
output_bytes = tf.identity(output, name='output_bytes')


def build_signatures(*tensor_list):
  signatures = {t.name: tf.saved_model.utils.build_tensor_info(
      t) for t in tensor_list}
  return signatures


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  builder = tf.saved_model.builder.SavedModelBuilder('/tmp/simple_img_model/1')

  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          sm.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          sm.signature_def_utils.build_signature_def(
              inputs=build_signatures(input_bytes),
              outputs=build_signatures(output_bytes),
              method_name=sm.signature_constants.PREDICT_METHOD_NAME)
      })

  builder.save()
