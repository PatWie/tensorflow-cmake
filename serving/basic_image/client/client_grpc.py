import tensorflow as tf
import grpc
import base64
import argparse

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


"""
pip install tensorflow-serving-api --user
"""


def main(fn):
  channel = grpc.insecure_channel("localhost:8500")

  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  request = predict_pb2.PredictRequest()

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'simple_img_model'
  request.model_spec.signature_name = 'serving_default'

  image_buffer = open(fn, "rb").read()

  request.inputs['input_bytes:0'].CopyFrom(
      tf.contrib.util.make_tensor_proto([image_buffer], shape=[1]))

  response = stub.Predict(request, timeout=5.0)
  response_b64 = response.outputs['output_bytes:0'].string_val[0]
  response_buffer = base64.urlsafe_b64decode(response_b64)
  open('prediction_grpc.png', 'wb').write(response_buffer)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('fn', help='path to img', type=str)
  args = parser.parse_args()
  main(args.fn)
