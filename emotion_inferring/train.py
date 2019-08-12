from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import sklearn as sk
import matplotlib.pyplot as plt

from datetime import datetime
from emotion_inferring.dataset.data_reader import DataReader
from emotion_inferring.model.model import Model_Creator
from .utils import *


def train(args, hparams):
  try:
    directories = validate_directories(args)
  except ValueError as e:
    print("Some arguments are wrong:")
    print(str(e))
    return

  logdir = directories['logdir']
  logdir = os.path.join(args.outputdir, args.database, logdir)
  os.makedirs(logdir, exist_ok=True)

  restore_from = directories['restore_from']
  restore_from = os.path.join(args.outputdir, args.database, restore_from)

  with tf.device('/cpu:0'):
    # Create coordinator.
    coord = tf.train.Coordinator()
    tf_global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate, optimizer = create_optimizer(tf_global_step, args, hparams)

    # Load raw waveform from corpus.
    with tf.name_scope('create_inputs'):
      reader = DataReader(hparams=hparams, logdir=logdir, coord=coord)
      input, text, target = reader.dequeue_train(args.batch_size)
      input_valid, text_valid, target_valid = reader.dequeue_valid(
          args.batch_size)

  # Create network.
  with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    net = Model_Creator(hparams=hparams)

    loss, acc, train_true, train_pred = net.inference(
        acoustic_features=input,
        textual_features=text,
        emotion_targets=target,
        target_class_dim=hparams.class_dim,
        l2_regularization_strength=hparams.L2_reg)

    grads = optimizer.compute_gradients(loss)

  with tf.device('/cpu:0'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)

  gradients = [grad for grad, var in grads]
  variables = [var for grad, var in grads]
  clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                   hparams.gradients_limit)
  loss_train_op = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                            global_step=tf_global_step)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  variable_averages = tf.train.ExponentialMovingAverage(0.999, tf_global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  ave_vars = [
      variable_averages.average(var) for var in tf.trainable_variables()
  ]
  train_op = tf.group(loss_train_op, update_ops, variables_averages_op)

  # Validation
  with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    loss_valid, acc_valid, valid_true, valid_pred, GCA_score = net.inference(
        acoustic_features=input_valid,
        textual_features=text_valid,
        emotion_targets=target_valid,
        target_class_dim=hparams.class_dim,
        l2_regularization_strength=hparams.L2_reg,
        is_training=False)

  # Start Training
  with tf.Session() as sess:
    # Run the initializer
    print("Initializing ... ")
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    summaries = tf.summary.merge_all()
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    var_list += ave_vars

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=var_list, max_to_keep=hparams.max_to_keep)

    try:
      saved_global_step = load(saver, sess, restore_from)
      if hparams.is_overwritten_training or saved_global_step is None:
        saved_global_step = -1

    except:
      print("Something went wrong while restoring checkpoint. "
            "We will terminate training to avoid accidentally overwriting "
            "the previous model.")
      raise

    saver = tf.train.Saver(var_list=tf.trainable_variables())
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads_train(sess)
    reader.start_threads_valid(sess)

    last_saved_step = saved_global_step
    tf_global_step = tf.assign(ref=tf_global_step,
                               value=tf.constant(last_saved_step))
    sess.run([tf_global_step])
    tf.contrib.slim.model_analyzer.analyze_vars(tf.trainable_variables(),
                                                print_info=True)

    Acc_save = 0
    f1_save = 0
    L_save, P_save = [1], [0]

    try:
      for step in range(saved_global_step + 1, args.num_steps):
        start_time = time.time()
        summary_value, loss_value, acc_value, _, lr, T_true, T_pred = \
            sess.run([summaries, loss, acc, train_op, learning_rate, train_true, train_pred])
        # writer.add_summary(summary_value, step)
        duration = time.time() - start_time

        print(
            'step {:d} - loss = {:.3f} acc = {:.3f}, lr = {:.6f} ({:.3f} sec/step)'
            .format(step, loss_value, acc_value, lr, duration))
        print("Accuracy", sk.metrics.accuracy_score(T_true, T_pred))

        if step % args.checkpoint_every == 1:
          ACC, F1, y_true, y_pred = validate(args.batch_size,
                                             sess,
                                             loss_valid,
                                             acc_valid,
                                             valid_true,
                                             valid_pred,
                                             GCA_score,
                                             summaries,
                                             writer,
                                             step,
                                             std_out=False)
          if ACC > Acc_save:
            Acc_save = ACC
            f1_save = F1
            L_save = y_true
            P_save = y_pred
            save(saver, sess, logdir, step)

          print("Best Accuracy", Acc_save)
          print("Precision",
                sk.metrics.precision_score(L_save, P_save, average='macro'))
          print("Recall",
                sk.metrics.recall_score(L_save, P_save, average='macro'))
          print("f1_score", f1_save)
          print("confusion_matrix")
          print(sk.metrics.confusion_matrix(L_save, P_save))

    finally:
      _, _, _, _ = validate(args.batch_size,
                            sess,
                            loss_valid,
                            acc_valid,
                            valid_true,
                            valid_pred,
                            GCA_score,
                            summaries,
                            writer,
                            step,
                            std_out=True)
      print("Best Accuracy", Acc_save)
      print("Precision",
            sk.metrics.precision_score(L_save, P_save, average='macro'))
      print("Recall", sk.metrics.recall_score(L_save, P_save, average='macro'))
      print("f1_score", f1_save)
      print("confusion_matrix")
      print(sk.metrics.confusion_matrix(L_save, P_save))
      coord.request_stop()
      coord.join(threads)


def validate(batch_size, sess, loss_op, acc_op, label_op, pred_op, GCA_score_op,
             summaries_op, writer, step, std_out):
  print(" Current performance is   ... ")
  eva_samples = 512
  if std_out:
    eva_samples = 512
  tower_acc, tower_loss, tower_label_true, tower_label_pred, summaries_pred = [], [], [], [], []
  for _ in range(int(eva_samples / batch_size)):
    loss_val, acc_val, y_true, y_pred, GCA_score, summaries_pred = \
        sess.run([loss_op, acc_op, label_op, pred_op, GCA_score_op, summaries_op])
    tower_loss.append(loss_val)
    tower_acc.append(acc_val)
    tower_label_true.append(y_true)
    tower_label_pred.append(y_pred)
  if std_out:
    writer.add_summary(summaries_pred, step)
  y_true = np.array(tower_label_true).reshape(-1)
  y_pred = np.array(tower_label_pred).reshape(-1)
  ACC = sk.metrics.accuracy_score(y_true, y_pred)
  print("Accuracy", ACC)
  print("Precision", sk.metrics.precision_score(y_true, y_pred,
                                                average='macro'))
  print("Recall", sk.metrics.recall_score(y_true, y_pred, average='macro'))
  F1 = sk.metrics.f1_score(y_true, y_pred, average='macro')
  print("f1_score", F1)
  print("confusion_matrix")
  print(sk.metrics.confusion_matrix(y_true, y_pred))
  if std_out:
    np.save('/home/runnan/ER_system/log/GCA_scores', GCA_score)
  return ACC, F1, y_true, y_pred


def er_train(args, hparams):
  return train(args, hparams)
