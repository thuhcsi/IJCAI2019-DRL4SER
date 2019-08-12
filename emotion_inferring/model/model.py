import numpy as np
import tensorflow as tf

from emotion_inferring.model.modules import *
from emotion_inferring.utils import *


class Model_Creator(object):

  def __init__(self, hparams):

    self.hparams = hparams
    self.batch_size = hparams.batch_size
    self.GRU_units = hparams.units
    self.Training = True

  def _create_network(self,
                      acoustic_features,
                      textual_features=None,
                      target_class_dim=None,
                      is_training=True,
                      presentation_output=False):

    temp = []
    ## acoustic feature extraction
    if self.hparams.acoustic_enable:
      if self.hparams.CNN_extractor:
        acoustic_features = Stack_CNN_blocks(acoustic_features,
                                             self.hparams.self_attention,
                                             is_training)

      ## Acoustic representation generation
      if self.hparams.global_attention:
        speech_representation, temp = GCA_LSTM(acoustic_features,
                                               self.hparams.units,
                                               self.hparams.GCA_iterations,
                                               'SPEECH', is_training)
      elif self.hparams.mixture_attention:
        speech_representation = MA_LSTM(acoustic_features, self.hparams.units,
                                        self.hparams.GCA_iterations, 'SPEECH')
      else:
        speech_representation = LSTM_2_layers(acoustic_features,
                                              self.hparams.units, 'SPEECH')
        speech_representation = speech_representation[:, -1]

    if self.hparams.text_enable:
      ## Textual representation generation
      textual_representation = BLSTM_2_layers(textual_features, 128, 'TEXT',
                                              is_training)
      textual_representation = textual_representation[:, -1]

    if self.hparams.acoustic_enable and self.hparams.text_enable:
      ## Representation Merge
      merged_representation = tf.concat(
          [speech_representation, textual_representation], axis=1)

    elif self.hparams.acoustic_enable:
      merged_representation = speech_representation
    elif self.hparams.text_enable:
      merged_representation = textual_representation
    else:
      print('A input is needed!!')
      exit()

    if presentation_output:
      return merged_representation
    else:
      # Dense layer
      pre_logits = output_layer(merged_representation, target_class_dim,
                                is_training)
      return pre_logits, temp[:, 2:]

  def inference(self,
                acoustic_features,
                textual_features,
                emotion_targets=None,
                target_class_dim=None,
                l2_regularization_strength=None,
                is_training=True,
                predicting=False):
    with tf.name_scope('ER_model'):
      if emotion_targets is not None:
        emotion_class = tf.reshape(emotion_targets, [-1, 1])
        emotion_class = one_hot(emotion_class, target_class_dim)
      else:
        acoustic_features = tf.expand_dims(acoustic_features, axis=0)
        textual_features = tf.expand_dims(textual_features, axis=0)
      pred_logits, temp = self._create_network(acoustic_features,
                                               textual_features,
                                               target_class_dim, is_training)
      if predicting:
        return pred_logits
      else:
        with tf.name_scope('loss_acc'):
          xent = tf.nn.softmax_cross_entropy_with_logits_v2(
              logits=[pred_logits], labels=[emotion_class], name="xent_raw")
          loss = tf.reduce_mean(xent)

          ori_class = tf.argmax(tf.nn.softmax(emotion_class), axis=-1)
          pre_class = tf.argmax(tf.nn.softmax(pred_logits), axis=1)
          acc = tf.metrics.accuracy(labels=ori_class, predictions=pre_class)[1]

          if not is_training:
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('acc', acc)
            return loss, acc, ori_class, pre_class, temp
          else:
            if l2_regularization_strength is None:
              tf.summary.scalar('loss', loss)
              tf.summary.scalar('acc', acc)
              return loss, acc, ori_class, pre_class
            else:
              # L2 regularization for all trainable parameters
              l2_loss = tf.add_n([
                  tf.nn.l2_loss(v)
                  for v in tf.trainable_variables()
                  if not ('bias' in v.name)
              ])
              # Add the regularization term to the loss
              total_loss = (loss + l2_regularization_strength * l2_loss)
              tf.summary.scalar('acc', acc)
              tf.summary.scalar('l2_loss', l2_loss)
              tf.summary.scalar('total_loss', total_loss)
              return total_loss, acc, ori_class, pre_class

  def representation(self, acoustic_features, textual_features):
    with tf.name_scope('ER_model'):
      acoustic_features = tf.expand_dims(acoustic_features, axis=0)
      textual_features = tf.expand_dims(textual_features, axis=0)
      speech_represenation = self._create_network(
          acoustic_features=acoustic_features,
          textual_features=textual_features,
          is_training=False,
          presentation_output=True)
    return speech_represenation
