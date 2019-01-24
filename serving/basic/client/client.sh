#!/usr/bin/env bash

curl -d '{"instances": [1.0]}' -X POST http://localhost:8501/v1/models/simple_model:predict