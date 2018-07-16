import tensorflow as tf
import sys
import os
from distutils.dir_util import copy_tree



pairs = [
    (os.path.join(os.environ['TensorFlow_GIT_REPO'], "bazel-tensorflow/tensorflow/cc"), "tensorflow/cc"),
    (os.path.join(os.environ['TensorFlow_GIT_REPO'], "bazel-genfiles/tensorflow/cc"), "tensorflow/cc")
]


for p in pairs:
    copy_tree(p[0], p[1])
    print("-- copied %s" % p[0])



# mkdir -p /graphics/opt/opt_Ubuntu16.04/tensorflow/build/v1.9.0/includes/tensorflow/cc/ops/
# cp /graphics/opt/opt_Ubuntu16.04/tensorflow/src/bazel-genfiles/tensorflow/cc/ops/*.h /graphics/opt/opt_Ubuntu16.04/tensorflow/build/v1.9.0/includes/tensorflow/cc/ops/