#!/usr/bin/env bash

docker run \
  -p 8501:8501 \
  -p 8500:8500 \
  --mount type=bind,source=/tmp/simple_model,target=/models/simple_model \
  -e MODEL_NAME=simple_model -t tensorflow/serving