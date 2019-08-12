import numpy as np
import tensorflow as tf
import time

class GAN_Train(object):
    def __init__(self,
                 model,
                 data_generator,
                 batch_size,
                 data_type,
                 data_base):

        self.model  = model
        self.reader = data_generator
        self.batch_size= batch_size
        self.data_type = data_type
        self.data_base = data_base
        self.two_const = tf.constant(2, dtype=None, shape=[batch_size, 1], name='two_const')
        self.tre_const = tf.constant(3, dtype=None, shape=[batch_size, 1], name='tre_const')
        # Emotional utterances are labeled as "1"
        self.one_const = tf.constant(1, dtype=None, shape=[batch_size, 1], name='one_const')

    def _one_hot(self, input_batch):
        with tf.name_scope('GAN_one_hot_encode'):
            depth = 2      # Same as the number of classes
            encoded = tf.one_hot(
                input_batch,
                depth=depth,
                dtype=tf.float32)
            shape = [-1, depth]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def _create_discriminator(self, input):
        with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
            Dense_1 = tf.contrib.layers.fully_connected(input, 1024)
            Dense_2 = tf.contrib.layers.fully_connected(Dense_1, 1024)
            Pre_class = tf.contrib.layers.fully_connected(Dense_2, 2)
        return Pre_class

    def _train_generator(self, input, optimizer,class_dim):
        label = self._one_hot(tf.to_int32(self.one_const))
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            GAN_input = self.model.representation(input, class_dim=class_dim)
        Discriminator_output = self._create_discriminator(GAN_input)
        with tf.variable_scope('Generator_train', reuse=tf.AUTO_REUSE):
            xent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=[Discriminator_output],
                                                              labels=[label],
                                                              name="xent_raw")
            acc_o = tf.argmax(tf.nn.softmax(label), axis=1)
            acc_p = tf.argmax(tf.nn.softmax(Discriminator_output), axis=1)
            acc = tf.metrics.accuracy(labels=acc_o, predictions=acc_p)[1]
            loss = tf.reduce_mean(xent)
            grads = optimizer.compute_gradients(loss, var_list= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'))
            gradients = [grad for grad, var in grads]
            variables = [var for grad, var in grads]
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.apply_gradients(zip(clipped_gradients, variables))
        return loss, acc, train_op

    def _train_discriminator(self, input, target, optimizer):
        target = tf.reshape(target, [-1, 1])
        if self.data_base == 'IEMOCAP':
            label  = tf.not_equal(target, self.two_const)
            label  = self._one_hot(tf.to_int32(label))
        elif self.data_base == 'RID':
            label  = tf.logical_or(tf.equal(target, self.two_const), tf.equal(target, self.tre_const))
            label  = tf.logical_not(label)
            label  = self._one_hot(tf.to_int32(label))
        else:
            label = None
            print('Wrong DATABASE!!')
            exit()
        Discriminator_output = self._create_discriminator(input)
        with tf.variable_scope('Discriminator_train', reuse=tf.AUTO_REUSE):
            xent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=[Discriminator_output],
                                                              labels=[label],
                                                              name="xent_raw")
            acc_o = tf.argmax(tf.nn.softmax(label), axis=1)
            acc_p = tf.argmax(tf.nn.softmax(Discriminator_output), axis=1)
            acc = tf.metrics.accuracy(labels=acc_o, predictions=acc_p)[1]
            loss = tf.reduce_mean(xent)
            grads = optimizer.compute_gradients(loss, var_list= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
            gradients = [grad for grad, var in grads]
            variables = [var for grad, var in grads]
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.apply_gradients(zip(clipped_gradients, variables))
        return loss, acc, train_op
