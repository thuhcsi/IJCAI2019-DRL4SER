from __future__ import division, print_function

import os, sys, re
import numpy as np
import tensorflow as tf
from scipy import stats
import soundfile as sf

from emotion_inferring.model.model import Model_Creator
from emotion_inferring.utils import *
from emotion_inferring.dataset.audio import acoustic_gen
from gensim.models.keyedvectors import KeyedVectors


class emotion_predictor(object):

  def __init__(self,
               hparams,
               checkpoint,
               sample_dir=None,
               logdir=None,
               is_training=False,
               presentation_output=False,
               convert_to_pb=False,
               pb_save_dir=None):

    self.hparams = hparams
    self.checkpoint = checkpoint
    self.logdir = logdir
    self.sample_dir = sample_dir
    self.is_training = is_training
    self.sample_rate = 16000

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
      with tf.device('/cpu:0'):
        self.net = Model_Creator(hparams=self.hparams)
        self.input_acous = tf.placeholder(dtype=tf.float32,
                                          shape=(None,
                                                 self.hparams.condition_num),
                                          name='input_acoustic_features')
        self.input_texts = tf.placeholder(dtype=tf.float32,
                                          shape=(None, 300),
                                          name='input_textual_features')
        if not presentation_output:
          self.predict_logits = self.net.inference(
              acoustic_features=self.input_acous,
              textual_features=self.input_texts,
              target_class_dim=4,
              is_training=False,
              predicting=True)
          self.predict_class = tf.nn.softmax(
              tf.cast(self.predict_logits, tf.float64))
          self.predict_class = tf.argmax(self.predict_class,
                                         axis=-1,
                                         name='predicted_class')
        elif presentation_output:
          self.generated_representation = self.net.representation(
              acoustic_features=self.input_acous,
              textual_features=self.input_texts)
          self.generated_representation = tf.reshape(
              self.generated_representation, [-1], name='speech_representation')
        else:
          raise Exception('MODE ERROR!')

    self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    self.sess.run(tf.global_variables_initializer())
    self.sess.run(tf.local_variables_initializer())
    self.saver = tf.train.Saver(var_list=tf.trainable_variables())
    self.load(self.saver, self.sess, self.checkpoint)

    if convert_to_pb:
      if not presentation_output:
        target_name = 'model/predicted_class'
        pb_name = "/emotion_class_inferring_frozen_model.pb"
      elif presentation_output:
        target_name = 'model/speech_representation'
        pb_name = "/speech_representation_generation_frozen_model.pb"

      target_save_dir = os.path.dirname(
          os.path.realpath(__file__)) + '/' + pb_save_dir
      os.makedirs(target_save_dir, exist_ok=True)
      output_grap = target_save_dir + pb_name
      print('Saving .PB in : ' + output_grap)
      output_grap_def = tf.graph_util.convert_variables_to_constants(
          self.sess,
          tf.get_default_graph().as_graph_def(),
          output_node_names=[target_name])
      with tf.gfile.GFile(output_grap, 'wb') as f:
        f.write(output_grap_def.SerializeToString())
      print("%d ops in the final graph." % len(output_grap_def.node))

    self.mel_min, self.mel_max = np.load(self.checkpoint +
                                         '/mel_min_max_var.npy')
    self.Word2Vec = KeyedVectors.load_word2vec_format(hparams.word2vec_path,
                                                      binary=True)

  def load(self, saver, sess, checkpoint):
    print("Trying to restore saved checkpoints from {} ...".format(checkpoint),
          end="")
    ckpt = tf.train.get_checkpoint_state(checkpoint)
    if ckpt:
      print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
      global_step = int(
          ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
      print("  Global step was: {}".format(global_step))
      print("  Restoring...", end="")
      saver.restore(sess, ckpt.model_checkpoint_path)
      print(" Done.")
      return global_step
    else:
      print(" No checkpoint found.")
      return None

  def inferring(self, filename):
    emotions_used = np.array(['ang', 'hap', 'neu', 'sad'])
    acous, text = self.input_gen(
        filename=os.path.join(self.sample_dir, filename))
    print('Inferring emotion in ' + filename + ' ......')
    output = self.sess.run(self.predict_class,
                           feed_dict={
                               self.input_acous: acous,
                               self.input_texts: text
                           })
    return emotions_used[output]

  def presentation(self, filename):
    acous, text = self.input_gen(
        filename=os.path.join(self.sample_dir, filename))
    print('Producing representation of ' + filename + ' ......')
    output = self.sess.run(self.generated_representation,
                           feed_dict={
                               self.input_acous: acous,
                               self.input_texts: text
                           })
    return output

  def input_gen(self, filename):
    audio, fs = sf.read(filename)
    acoustic_features = acoustic_gen(self.hparams,
                                     audio,
                                     mel_max=self.mel_max,
                                     mel_min=self.mel_min)
    trans_file = open(filename[:-4] + '.txt', 'r').read()
    trans_file = np.array(trans_file.split('\n'))
    transcriptions = re.split(r' ', str(trans_file))
    transcriptions_emb = []
    for word in transcriptions:
      word = ''.join(filter(str.isalpha, word))
      transcriptions_emb.append(np.array(self.Word2Vec[word]))
    textual_features = np.asarray(transcriptions_emb)
    return acoustic_features, textual_features
