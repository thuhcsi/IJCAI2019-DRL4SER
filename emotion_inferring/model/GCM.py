import tensorflow as tf
from tensorflow.python.ops import *


class GCAttention(tf.contrib.seq2seq.AttentionMechanism):

  def __init__(self, init_F, name="GCAttention"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      self.batch_size = init_F.get_shape().as_list()[0]
      self._num_units = init_F.get_shape().as_list()[1]
      self._name = name
      self.WF = tf.get_variable(
          "WF",
          shape=[2 * self._num_units, self._num_units],
          initializer=tf.contrib.layers.xavier_initializer(),
          trainable=True)
      self.GCA_memory = init_F

  def score(self, inputs):
    with tf.variable_scope('SCORE', reuse=tf.AUTO_REUSE):
      GCAcell = GCACell(self._num_units)
      scores, _ = tf.nn.dynamic_rnn(cell=GCAcell,
                                    inputs=inputs,
                                    initial_state=self.GCA_memory,
                                    dtype=tf.float32,
                                    scope='score_computing')
    return tf.nn.softmax(scores, axis=-2)

  def update(self, new_F):
    with tf.variable_scope('UPDATE', reuse=tf.AUTO_REUSE):
      self.GCA_memory = tf.nn.relu(
          math_ops.matmul(tf.concat([new_F, self.GCA_memory], -1), self.WF))

  def output(self):
    return self.GCA_memory


class GCACell(tf.nn.rnn_cell.RNNCell):

  def __init__(self, output_size, activation=tf.tanh, reuse=tf.AUTO_REUSE):
    super(GCACell, self).__init__(_reuse=reuse)
    self._num_units = output_size
    self._activation = activation

    self.eW1 = tf.get_variable(
        "eW1",
        shape=[2 * self._num_units, self._num_units],
        initializer=tf.contrib.layers.xavier_initializer())
    self.eW2 = tf.get_variable(
        "eW2",
        shape=[self._num_units, 1],
        initializer=tf.contrib.layers.xavier_initializer())

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return 1

  def call(self, inputs, state):
    with tf.variable_scope('SCORE_COMPUTING', reuse=tf.AUTO_REUSE):
      e = math_ops.matmul(
          self._activation(
              math_ops.matmul(array_ops.concat([inputs, state], 1), self.eW1)),
          self.eW2)
    return e, state
