# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modifications Copyright 2017 Arm Inc. All Rights Reserved. 
# Added new model definitions for speech command recognition used in
# the paper: https://arxiv.org/pdf/1711.07128.pdf
# Added Quantized model definitions 
#

"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }


def create_model(fingerprint_input, model_settings, model_architecture,
                 model_size_info, act_max, is_training, \
                 runtime_settings=None):
  """Builds a model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible model architectures that can be
  requested.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  if model_architecture == 'dnn':
    return create_dnn_model(fingerprint_input, model_settings, model_size_info,
                              act_max, is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv", "low_latency_svdf",'+ 
                    ' "dnn", "cnn", "basic_lstm", "lstm",'+
                    ' "gru", "crnn" or "ds_cnn"')


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def create_dnn_model(fingerprint_input, model_settings, model_size_info, 
                       act_max, is_training):
  """Builds a model with multiple hidden fully-connected layers.
  model_size_info: length of the array defines the number of hidden-layers and
                   each element in the array represent the number of neurons 
                   in that layer 
  """

  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  num_layers = len(model_size_info)
  layer_dim = [fingerprint_size]
  layer_dim.extend(model_size_info)
  flow = fingerprint_input
  if(act_max[0]!=0):
    flow = tf.fake_quant_with_min_max_vars(flow, min=-act_max[0], \
               max=act_max[0]-(act_max[0]/128.0), num_bits=8)
  for i in range(1, num_layers + 1):
      with tf.variable_scope('fc'+str(i)):
          W = tf.get_variable('W', shape=[layer_dim[i-1], layer_dim[i]], 
                initializer=tf.contrib.layers.xavier_initializer())
          b = tf.get_variable('b', shape=[layer_dim[i]])
          flow = tf.matmul(flow, W) + b
          if(act_max[i]!=0):
            flow = tf.fake_quant_with_min_max_vars(flow, min=-act_max[i], \
                       max=act_max[i]-(act_max[i]/128.0), num_bits=8)
          flow = tf.nn.relu(flow)
          if is_training:
            flow = tf.nn.dropout(flow, dropout_prob)

  weights = tf.get_variable('final_fc', shape=[layer_dim[-1], label_count], 
              initializer=tf.contrib.layers.xavier_initializer())
  bias = tf.Variable(tf.zeros([label_count]))
  logits = tf.matmul(flow, weights) + bias
  if(act_max[num_layers+1]!=0):
    logits = tf.fake_quant_with_min_max_vars(logits, min=-act_max[num_layers+1], \
                 max=act_max[num_layers+1]-(act_max[num_layers+1]/128.0), num_bits=8)
  if is_training:
    return logits, dropout_prob
  else:
    return logits

