import numpy as np
import tensorflow as tf
import sys

from emotion_inferring.model.GCALSTM import GCALSTMCell
from emotion_inferring.model.MALSTM import MALSTMCell
from emotion_inferring.model.GCM import GCAttention
from emotion_inferring.model.SelfAttention import *

def Creat_CNN_layer(input, filter_num, kernel_size, strides=1, activation=tf.nn.relu, padding='same', dilation_rate=1,
                    dropout=0.5, Training=True, pooling=False, pool_size=None, pool_stride=None):

    conv = tf.layers.conv1d(inputs=input, filters=filter_num, kernel_size=kernel_size,
                            strides=strides, activation = activation, padding = padding, dilation_rate=dilation_rate)
    conv = tf.layers.batch_normalization(conv, training=Training)
    conv = activation(conv)
    out = tf.layers.dropout(conv, rate=dropout, training=Training)
    if pooling:
        out = tf.layers.max_pooling1d(inputs=out, pool_size=pool_size,
                                      strides=pool_stride, padding=padding)
    return out

def self_attention_block(input, heads, forget, training, add_op = True, reverse=False, Record = False):
    SeA_layer = SelfAttention(input.get_shape().as_list()[2], heads, forget, training)
    attention_bias = get_padding_bias(input[:, :, -1])
    if (not training) and Record:
        score = SeA_layer(input, attention_bias, reverse=reverse, record = Record)
    else:
        score = SeA_layer(input, attention_bias, reverse=reverse)
    if add_op:
        SeA_out = tf.add(score, input)
    else:
        SeA_out = score
    return SeA_out

def CNN_block(input, filter_num, kernel_size, SelfAttention = True,
              Training=True, padding = 'same', stride_op = False, record = False):
    if stride_op:
        conv_input = tf.layers.conv1d(inputs=input, filters=filter_num, kernel_size=kernel_size, strides=2)
    else:
        conv_input = tf.layers.conv1d(inputs=input, filters=filter_num, kernel_size=kernel_size, strides=1)

    conv = Creat_CNN_layer(conv_input, filter_num, kernel_size, activation= tf.nn.relu, padding= padding,
                           Training=Training, pooling=False)
    if SelfAttention:
        conv = self_attention_block(conv, 4, 0.1, Training)

    conv = Creat_CNN_layer(conv, filter_num, kernel_size=kernel_size, activation= tf.nn.relu, padding= padding,
                           Training=Training, pooling=False)
    conv_mid = tf.add(conv_input, conv)

    conv = Creat_CNN_layer(conv_mid, filter_num, kernel_size, activation= tf.nn.relu, padding= padding,
                           Training=Training, pooling=False)
    if SelfAttention:
        conv = self_attention_block(conv, 4, 0.1, Training, Record= record)
    conv = Creat_CNN_layer(conv, filter_num, kernel_size=kernel_size, activation= tf.nn.relu, padding= padding,
                           Training=Training, pooling=False)
    conv = tf.add(conv_mid, conv)
    return conv

def Stack_CNN_blocks(input, SelfAttention = True, Training = True):
    # For Acoustic features, the window length is 50ms, and shift is 10ms
    # Downsampling to window_size = 160ms, window_shift = 20ms
    conv_input = tf.layers.conv1d(inputs=input, filters=32, kernel_size=10, strides=2)
    # Downsampling to window_size = 480ms, window_shift = 40ms
    conv_1 = CNN_block(conv_input,  filter_num = 64,  kernel_size = 5, SelfAttention = False,
                       Training=Training, stride_op = True, record = True)
    # Downsampling to window_size = 1440ms, window_shift = 80ms
    conv_2 = CNN_block(conv_1, filter_num = 128,  kernel_size = 5, SelfAttention = SelfAttention,
                       Training=Training, stride_op = True, record = True)
    conv_3 = CNN_block(conv_2, filter_num = 256,  kernel_size = 3, SelfAttention = SelfAttention,
                       Training=Training, stride_op = False, record = True)
    conv_4 = CNN_block(conv_3, filter_num = 512,  kernel_size = 3, SelfAttention = SelfAttention,
                       Training=Training, stride_op = False, record = True)

    # if Training == False:
    #     conv_1_output = tf.expand_dims(conv_1, axis= -1)
    #     tf.summary.image('conv_1_output', conv_1_output, max_outputs=3, collections=None, family=None)
    #     conv_2_output = tf.expand_dims(conv_2, axis= -1)
    #     tf.summary.image('conv_2_output', conv_2_output, max_outputs=3, collections=None, family=None)
    #     conv_3_output = tf.expand_dims(conv_3, axis= -1)
    #     tf.summary.image('conv_3_output', conv_3_output, max_outputs=3, collections=None, family=None)
    #     conv_4_output = tf.expand_dims(conv_4, axis= -1)
    #     tf.summary.image('conv_4_output', conv_4_output, max_outputs=3, collections=None, family=None)
    return conv_4


def lstm_cell(cells):
    return tf.contrib.rnn.BasicLSTMCell(cells)

def LSTM_2_layers(inputs, units, name):
    LSTM_layer_1 = lstm_cell(units)
    LSTM_layer_2 = lstm_cell(units)
    outputs_layer_1, _ = tf.nn.dynamic_rnn(cell  =LSTM_layer_1,
                                           inputs=inputs,
                                           dtype =tf.float32,
                                           scope = name + '-LSTM-Layer-1')
    outputs_layer_1    = tf.contrib.layers.layer_norm(outputs_layer_1)
    outputs_layer_2, _ = tf.nn.dynamic_rnn(cell  =LSTM_layer_2,
                                           inputs=outputs_layer_1,
                                           dtype =tf.float32,
                                           scope = name + '-LSTM-Layer-2')
    return outputs_layer_2

def GCA_LSTM(inputs, units, iterations, name, training):
    LSTM_layer_1 = lstm_cell(units)
    LSTM_layer_2 = GCALSTMCell(units)
    outputs_layer_1, _ = tf.nn.dynamic_rnn(cell  =LSTM_layer_1,
                                            inputs=inputs,
                                            dtype =tf.float32,
                                            scope =name+ '-GCA-LSTM-Layer-1')
    outputs_layer_1 = tf.contrib.layers.layer_norm(outputs_layer_1)
    GCAmemory = GCAttention(outputs_layer_1[:, -1])
    temp = outputs_layer_1[:, :, -2:]
    for _ in range(iterations):
        score = GCAmemory.score(outputs_layer_1)
        score = tf.divide(score - tf.reduce_min(score, axis=1, keepdims= True),
                          tf.reduce_max(score, axis=1, keepdims= True) - tf.reduce_min(score, axis=1, keepdims= True))
        temp  = tf.concat([temp, score], -1)
        LSTM_layer_2_input = tf.concat([outputs_layer_1, score], -1)
        outputs_layer_2, _ = tf.nn.dynamic_rnn(cell  =LSTM_layer_2,
                                               inputs=LSTM_layer_2_input,
                                               dtype =tf.float32,
                                               scope = name+ '-GCA-LSTM-Layer-2')
        GCAmemory.update(outputs_layer_2[:, -1])
    return GCAmemory.output(), temp


def MA_LSTM(inputs, units, iterations, name):
    LSTM_layer_1 = lstm_cell(units)
    LSTM_layer_2 = MALSTMCell(units)
    outputs_layer_1, _ = tf.nn.dynamic_rnn(cell  =LSTM_layer_1,
                                           inputs=inputs,
                                           dtype =tf.float32,
                                           scope =name+ '-MA-LSTM-Layer-1')
    outputs_layer_1 = tf.contrib.layers.layer_norm(outputs_layer_1)
    GCAmemory = GCAttention(outputs_layer_1[:, -1])
    inputs_layer_2 = outputs_layer_1

    for _ in range(iterations):
        score = GCAmemory.score(outputs_layer_1)
        self_attention_h1 = self_attention_block(inputs_layer_2, 4, 0.1, training=True)
        LSTM_layer_2_input = tf.concat([inputs_layer_2, self_attention_h1, score], -1)
        outputs_layer_2, _ = tf.nn.dynamic_rnn(cell  = LSTM_layer_2,
                                               inputs= LSTM_layer_2_input,
                                               dtype = tf.float32,
                                               scope = name+ '-MA-LSTM-Layer-2')
        GCAmemory.update(outputs_layer_2[:, -1])
        inputs_layer_2 = outputs_layer_2
    return GCAmemory.output()


def BLSTM_2_layers(inputs, units, name, is_training):
    BLSTM_layer_1_f = lstm_cell(units)
    BLSTM_layer_1_b = lstm_cell(units)
    BLSTM_layer_2_f = lstm_cell(units)
    BLSTM_layer_2_b = lstm_cell(units)

    normalized_input = tf.layers.batch_normalization(inputs, training = is_training)
    outputs_layer_1, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=BLSTM_layer_1_f,
                                                         cell_bw=BLSTM_layer_1_b,
                                                         inputs =normalized_input,
                                                         scope  = name + '-BLSTM-Layer-1',
                                                         dtype  =tf.float32)
    outputs_layer_1 = tf.concat(outputs_layer_1, 2)
    outputs_layer_1 = tf.contrib.layers.layer_norm(outputs_layer_1)
    outputs_layer_2, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=BLSTM_layer_2_f,
                                                         cell_bw=BLSTM_layer_2_b,
                                                         inputs =outputs_layer_1,
                                                         scope  = name + '-BLSTM-Layer-2',
                                                         dtype  =tf.float32)
    outputs_layer_2 = tf.concat(outputs_layer_2, 2)
    return  outputs_layer_2

def output_layer(input, target_dim, Training, dropout=0.5, units=256):
    Dense_1 = tf.contrib.layers.fully_connected(input, units)
    Dense_drop = tf.layers.dropout(Dense_1, rate=dropout, training=Training)
    Dense_2 = tf.contrib.layers.fully_connected(Dense_drop, units)
    pre_logits = tf.contrib.layers.fully_connected(Dense_2, target_dim)
    return pre_logits