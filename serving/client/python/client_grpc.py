import tensorflow as tf
import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


"""
pip install tensorflow-serving-api --user
"""

data = 1.0
timeout = 5.0  # seconds

channel = grpc.insecure_channel("localhost:8500")

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()

request = predict_pb2.PredictRequest()
request.model_spec.name = 'simple_model'
request.model_spec.signature_name = 'serving_default'

request.inputs['input'].CopyFrom(
    tf.contrib.util.make_tensor_proto([data], shape=[1]))

print(stub.Predict(request, timeout))
