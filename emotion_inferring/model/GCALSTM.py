import collections
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class GCALSTMCell(tf.nn.rnn_cell.RNNCell):
  def __init__(self,
               num_units,
               activation=None,
               reuse=tf.AUTO_REUSE,
               name='GCALSTMCell',
               dtype=None):

    super(GCALSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)
    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)
    self._num_units = num_units
    self._activation= activation or math_ops.tanh

  @property
  def state_size(self):
    return LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)
    input_depth = inputs_shape[1].value - 1
    h_depth = self._num_units
    self._kernel = self.add_variable(
      _WEIGHTS_VARIABLE_NAME,
      shape=[input_depth + h_depth, 4 * self._num_units])
    self._bias = self.add_variable(
      _BIAS_VARIABLE_NAME,
      shape=[4 * self._num_units],
      initializer=init_ops.zeros_initializer(dtype=self.dtype))

  def call(self, inputs, state):
    sigmoid = math_ops.sigmoid
    one = constant_op.constant(1, dtype=dtypes.int32)
    x = inputs[:,:-1]
    s = inputs[:,-1]
    c, h = state
    one_tensor = constant_op.constant(1, dtype=self.dtype)
    gate_inputs = math_ops.matmul(array_ops.concat([x, h], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

    i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)
    add = math_ops.add
    multiply = math_ops.multiply
    sub = math_ops.subtract
    new_c = add(multiply(multiply(c, sigmoid(f)), tf.expand_dims(sub(one_tensor,s),-1)),
                multiply(multiply(sigmoid(i), self._activation(j)), tf.expand_dims(s,-1)))
    new_h = multiply(self._activation(new_c), sigmoid(o))
    new_state = LSTMStateTuple(new_c, new_h)
    return new_h, new_state

  def GCAupdate(self, GCA_value):
    tf.assign(self.GCA_memory, GCA_value)