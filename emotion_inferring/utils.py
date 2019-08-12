from __future__ import division, print_function

import tensorflow as tf
import wave, os, sys
import soundfile as sf
import numpy as np
import librosa

from datetime import datetime


def create_adam_optimizer(learning_rate, momentum):
  return tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-6)


def create_sgd_optimizer(learning_rate, momentum):
  return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                    momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
  return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                   momentum=momentum,
                                   epsilon=1e-5)


optimizer_factory = {
    'adam': create_adam_optimizer,
    'sgd': create_sgd_optimizer,
    'rmsprop': create_rmsprop_optimizer
}


def save(saver, sess, logdir, step):
  model_name = 'model.ckpt'
  checkpoint_path = os.path.join(logdir, model_name)
  print('Storing checkpoint to {} ...'.format(logdir), end='')
  sys.stdout.flush()
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  saver.save(sess, checkpoint_path, global_step=step)
  print('Done.')


def load(saver, sess, logdir):
  print("Trying to restore saved checkpoints from {} ...".format(logdir),
        end='')
  ckpt = tf.train.get_checkpoint_state(logdir)
  if ckpt:
    print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
    global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    print("  Global step was: {}".format(global_step))
    print("  Restoring...", end="")
    saver.restore(sess, ckpt.model_checkpoint_path)
    print(" Done.")
    return global_step
  else:
    print(" No checkpoint found.")
    return None


def get_default_logdir(logdir_root, started_time):
  logdir = os.path.join(logdir_root, 'train', started_time)
  return logdir


def validate_directories(args):
  # Arrangement
  logdir_root = args.logdir_root
  if logdir_root is None:
    logdir_root = './logdir'

  logdir = args.logdir
  if logdir is None:
    logdir = get_default_logdir(logdir_root,
                                "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now()))
    print('Using default logdir: {}'.format(logdir))

  restore_from = args.restore_from
  if restore_from is None:
    restore_from = logdir

  return {
      'logdir': logdir,
      'logdir_root': args.logdir_root,
      'restore_from': restore_from
  }


def learning_rate_decay(global_step, args, hparams):
  warm_up_step = int(hparams.warm_up_step)
  decay_step = int(hparams.decay_step)
  learning_rate = tf.cond(
      global_step < warm_up_step,
      lambda: tf.convert_to_tensor(args.learning_rate),
      lambda: tf.train.exponential_decay(args.learning_rate, global_step -
                                         warm_up_step + 1, decay_step, 0.5))
  return tf.maximum(hparams.mini_lr, learning_rate)


def create_optimizer(global_step, args, hparams):
  with tf.variable_scope('optimizer'):
    learning_rate_decayed = learning_rate_decay(global_step, args, hparams)
    optimizer = optimizer_factory[args.optimizer](
        learning_rate=learning_rate_decayed, momentum=args.momentum)
  return learning_rate_decayed, optimizer


def one_hot(samples, channels):
  with tf.name_scope('one_hot_encode'):
    encoded = tf.one_hot(samples, depth=channels, dtype=tf.float32)
    shape = [samples.get_shape().as_list()[0], -1, channels]
    encoded = tf.reshape(encoded, shape)
  return encoded
