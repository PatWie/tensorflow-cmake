#!/usr/bin/env bash

docker run \
  -p 8501:8501 \
  -p 8500:8500 \
  --mount type=bind,source=/tmp/simple_img_model,target=/models/simple_img_model \
  -e MODEL_NAME=simple_img_model -t tensorflow/serving
