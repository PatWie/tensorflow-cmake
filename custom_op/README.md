Custom Op for TensorFlow
========================

This is a very simple example on adding custom C++/CUDA ops to TensorFlow and its intended usage is just being a starting point for other custom TensorFlow operations.

The current version is tested on TensorFlow v1.9. Run the script

```console
user@host $ pip install tensorflow-gpu --user # just the pip package is needed
user@host $ cd custom_op/user_ops
user@host $ cmake .

-- Detecting TensorFlow info
-- Detecting TensorFlow info - done
-- Found TensorFlow: (found appropriate version "1.10.0")
-- TensorFlow-ABI is 0
-- TensorFlow-INCLUDE_DIR is /home/user/.local/lib/python2.7/site-packages/tensorflow/include
-- TensorFlow-LIBRARY is /home/user/.local/lib/python2.7/site-packages/tensorflow/libtensorflow_framework.so
-- No TensorFlow-CC-LIBRARY detected
-- No TensorFlow source repository detected
-- will build custom TensorFlow operation "matrix_add" (CPU+GPU)
-- Configuring done
-- Generating done
-- Build files have been written to: /home/user/git/github.com/user/tensorflow-cmake/custom_op/user_ops


user@host $ make
user@host $ TF_CPP_MIN_LOG_LEVEL=3 python test_matrix_add.py

Ran 7 tests in 0.639s
OK


user@host $ cd ..
user@host $ python example.py

[[[[41.523315  41.72658   36.96695   60.987278 ]
   [33.65208   36.74169   51.775063  68.19713  ]
   [37.170284  36.84849   45.591274  25.134266 ]]

  [[46.50826    3.2704964 11.46896   38.345737 ]
   [39.256355  46.856155  31.763277  24.865925 ]
   [41.87127   32.543793  38.068893  38.32374  ]]]]

```