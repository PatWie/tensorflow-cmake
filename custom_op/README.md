Custom Op for TensorFlow
========================

This is a very simple example on adding custom C++/CUDA ops to TensorFlow and its intended usage is just being a starting point for other custom TensorFlow operations.

The current version is tested on TensorFlow v1.9. Run the script

```bash
python configure.py
cmake .
make
python test_matrix_add.py
```
