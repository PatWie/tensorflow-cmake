# 2018, Patrick Wieschollek <mail@patwie.com>

# manually generated file
import tensorflow as tf
import os
from tensorflow.python.framework import ops

__all__ = []

def load_op(name, has_grad=False):
  """Load operation and add it to __all__ for imports.

  Args:
      name (str): name of operation without "_op" suffix
      has_grad (bool, optional): gradient (if exists) should be loaded as well

  Returns:
      functions
  """
  global __all__
  path = os.path.join(os.path.dirname(__file__), '%s_op.so' % name)
  _module = tf.load_op_library(path)
  if has_grad:
    __all__.append('%s' % name)
    __all__.append('%s_grad' % name)
    return getattr(_module, '%s' % name), getattr(_module, '%s_grad' % name)
  else:
    __all__.append('%s' % name)
    return getattr(_module, '%s' % name)


matrix_add, matrix_add_grad = load_op('matrix_add', has_grad=True)


@ops.RegisterGradient("MatrixAdd")
def _MatrixAddGrad(op, *grads):
  bias = op.get_attr('bias')
  matA = op.inputs[0]
  matB = op.inputs[1]
  # top = op.outputs[0]
  topdiff = grads[0]
  return matrix_add_grad(matA, matB, topdiff, bias=bias)
