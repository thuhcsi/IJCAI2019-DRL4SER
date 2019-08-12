import fnmatch
import os,io
import random
import re
import threading

import soundfile as sf
import librosa
import numpy as np
import tensorflow as tf
import time

from scipy import stats
from emotion_inferring.dataset.audio import *

class DataReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''
    def __init__(self,
                 hparams,
                 coord,
                 logdir,
                 renew = False,
                 queue_size = 1024):

        self.threads = []
        self.hparams = hparams
        self.coord = coord
        self.padding = hparams.padding_len
        self.train_data, self.valid_data = load_metadata(hparams, renew = renew)

        input_dim = hparams.condition_num
        print("The feature dim of each time step is :", hparams.condition_num)

        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(self.padding, input_dim))
        self.text_placeholder  = tf.placeholder(dtype=tf.float32, shape=(None, 300))
        self.class_placeholder = tf.placeholder(dtype=tf.int32, shape=(1, 1))

        self.queue_train = tf.PaddingFIFOQueue(queue_size,
                                               dtypes=['float32', 'float32', 'int32'],
                                               shapes=[(self.padding, input_dim), (None,300), (1, 1)])
        self.enqueue_train = self.queue_train.enqueue([self.input_placeholder,
                                                       self.text_placeholder,
                                                       self.class_placeholder])
        self.queue_valid = tf.PaddingFIFOQueue(1024,
                                               dtypes=['float32', 'float32', 'int32'],
                                               shapes=[(self.padding, input_dim), (None,300), (1, 1)])
        self.enqueue_valid = self.queue_valid.enqueue([self.input_placeholder,
                                                       self.text_placeholder,
                                                       self.class_placeholder])

        normalization_initial(self.train_data, self.hparams, logdir)

        if self.padding != None:
            self.generator_train = aoustic_features_generator(self.train_data, self.hparams, self.padding)
            self.generator_valid = aoustic_features_generator(self.valid_data, self.hparams, self.padding)
        else:
            self.generator_train = aoustic_features_generator(self.train_data, self.hparams)
            self.generator_valid = aoustic_features_generator(self.valid_data, self.hparams)


    def dequeue_train(self, num_elements):
        output = self.queue_train.dequeue_many(num_elements)
        return output

    def dequeue_valid(self, num_elements):
        output = self.queue_valid.dequeue_many(num_elements)
        return output

    def thread_main_train(self, sess):
        while 1:
            input_batch, text_batch, class_batch = self.generator_train.__next__()
            sess.run(self.enqueue_train,
                     feed_dict={self.input_placeholder: input_batch,
                                self.text_placeholder: text_batch,
                                self.class_placeholder: class_batch})

    def thread_main_valid(self, sess):
        while 1:
            input_batch, text_batch, class_batch = self.generator_valid.__next__()
            sess.run(self.enqueue_valid,
                     feed_dict={self.input_placeholder: input_batch,
                                self.text_placeholder: text_batch,
                                self.class_placeholder: class_batch})

    def start_threads_train(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread_train = threading.Thread(target=self.thread_main_train, args=(sess,))
            thread_train.daemon = True
            thread_train.start()
            self.threads.append(thread_train)
        return self.threads

    def start_threads_valid(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread_valid = threading.Thread(target=self.thread_main_valid, args=(sess,))
            thread_valid.daemon = True
            thread_valid.start()
            self.threads.append(thread_valid)
        return self.threads