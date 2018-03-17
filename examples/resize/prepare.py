import tensorflow as tf
import sys
import os
from distutils.dir_util import copy_tree



if '__cxx11_abi_flag__' not in dir(tf):
    print("-- Cannot find the ABI version of TensorFlow.")
    print("-- Your TensorFlow version is probably too old. Please upgrade to at least TF v1.5.")
    sys.exit(1)

if os.environ['TensorFlow_GIT_REPO'] == "":
    print("-- You need to set the environment-variable TensorFlow_GIT_REPO pointing to the TF-source")
    sys.exit(1)

with open("tensorflow_config.txt", "w") as f:
    print("-- TensorFlow_ABI: {}".format(tf.__cxx11_abi_flag__))
    f.write("set(TensorFlow_ABI %i)\n" % tf.__cxx11_abi_flag__)
    print("-- TensorFlow_INCLUDE_DIRS: {}".format(tf.sysconfig.get_include()))
    f.write("set(TensorFlow_INCLUDE_DIRS \"%s\")\n" % tf.sysconfig.get_include())


pairs = [
    (os.path.join(os.environ['TensorFlow_GIT_REPO'], "bazel-tensorflow/tensorflow/cc"), "tensorflow/cc"),
    (os.path.join(os.environ['TensorFlow_GIT_REPO'], "bazel-genfiles/tensorflow/cc"), "tensorflow/cc")
]


for p in pairs:
    copy_tree(p[0], p[1])
    print("-- copied %s" % p[0])