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
#
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
                 model_size_info, is_training, runtime_settings=None):
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
  if model_architecture == 'single_fc':
    return create_single_fc_model(fingerprint_input, model_settings,
                                  is_training)
  elif model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'low_latency_conv':
    return create_low_latency_conv_model(fingerprint_input, model_settings,
                                         is_training)
  elif model_architecture == 'low_latency_svdf':
    return create_low_latency_svdf_model(fingerprint_input, model_settings,
                                         is_training, runtime_settings)
  elif model_architecture == 'dnn':
    return create_dnn_model(fingerprint_input, model_settings, model_size_info,
                              is_training)
  elif model_architecture == 'cnn':
    return create_cnn_model(fingerprint_input, model_settings, model_size_info,
                              is_training)
  elif model_architecture == 'basic_lstm':
    return create_basic_lstm_model(fingerprint_input, model_settings, 
                                     model_size_info, is_training)
  elif model_architecture == 'lstm':
    return create_lstm_model(fingerprint_input, model_settings, 
                               model_size_info, is_training)
  elif model_architecture == 'gru':
    return create_gru_model(fingerprint_input, model_settings, model_size_info,
                              is_training)
  elif model_architecture == 'crnn':
    return create_crnn_model(fingerprint_input, model_settings, model_size_info, 
                               is_training)
  elif model_architecture == 'ds_cnn':
    return create_ds_cnn_model(fingerprint_input, model_settings, 
                                 model_size_info, is_training)
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


def create_single_fc_model(fingerprint_input, model_settings, is_training):
  """Builds a model with a single hidden fully-connected layer.

  This is a very simple model with just one matmul and bias layer. As you'd
  expect, it doesn't produce very accurate results, but it is very fast and
  simple, so it's useful for sanity testing.

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  weights = tf.Variable(
      tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
  bias = tf.Variable(tf.zeros([label_count]))
  logits = tf.matmul(fingerprint_input, weights) + bias
  if is_training:
    return logits, dropout_prob
  else:
    return logits


def create_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.

  This is roughly the network labeled as 'cnn-trad-fpool3' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces fairly good quality results, but can involve a large number of
  weight parameters and computations. For a cheaper alternative from the same
  paper with slightly less accuracy, see 'low_latency_conv' below.

  During training, dropout nodes are introduced after each relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                             'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_conv_element_count, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_low_latency_conv_model(fingerprint_input, model_settings,
                                  is_training):
  """Builds a convolutional model with low compute requirements.

  This is roughly the network labeled as 'cnn-one-fstride4' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces slightly lower quality results than the 'conv' model, but needs
  fewer weight parameters and computations.

  During training, dropout nodes are introduced after the relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = input_time_size
  first_filter_count = 186
  first_filter_stride_x = 1
  first_filter_stride_y = 1
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
      1, first_filter_stride_y, first_filter_stride_x, 1
  ], 'VALID') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  first_conv_output_width = math.floor(
      (input_frequency_size - first_filter_width + first_filter_stride_x) /
      first_filter_stride_x)
  first_conv_output_height = math.floor(
      (input_time_size - first_filter_height + first_filter_stride_y) /
      first_filter_stride_y)
  first_conv_element_count = int(
      first_conv_output_width * first_conv_output_height * first_filter_count)
  flattened_first_conv = tf.reshape(first_dropout,
                                    [-1, first_conv_element_count])
  first_fc_output_channels = 128
  first_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_conv_element_count, first_fc_output_channels], stddev=0.01))
  first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
  first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 128
  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
  second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_fc_output_channels, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_low_latency_svdf_model(fingerprint_input, model_settings,
                                  is_training, runtime_settings):
  """Builds an SVDF model with low compute requirements.

  This is based in the topology presented in the 'Compressing Deep Neural
  Networks using a Rank-Constrained Topology' paper:
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
        [SVDF]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This model produces lower recognition accuracy than the 'conv' model above,
  but requires fewer weight parameters and, significantly fewer computations.

  During training, dropout nodes are introduced after the relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    The node is expected to produce a 2D Tensor of shape:
      [batch, model_settings['dct_coefficient_count'] *
              model_settings['spectrogram_length']]
    with the features corresponding to the same time slot arranged contiguously,
    and the oldest slot at index [:, 0], and newest at [:, -1].
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
      ValueError: If the inputs tensor is incorrectly shaped.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']

  # Validation.
  input_shape = fingerprint_input.get_shape()
  if len(input_shape) != 2:
    raise ValueError('Inputs to `SVDF` should have rank == 2.')
  if input_shape[-1].value is None:
    raise ValueError('The last dimension of the inputs to `SVDF` '
                     'should be defined. Found `None`.')
  if input_shape[-1].value % input_frequency_size != 0:
    raise ValueError('Inputs feature dimension %d must be a multiple of '
                     'frame size %d', fingerprint_input.shape[-1].value,
                     input_frequency_size)

  # Set number of units (i.e. nodes) and rank.
  rank = 2
  num_units = 1280
  # Number of filters: pairs of feature and time filters.
  num_filters = rank * num_units
  # Create the runtime memory: [num_filters, batch, input_time_size]
  batch = 1
  memory = tf.Variable(tf.zeros([num_filters, batch, input_time_size]),
                       trainable=False, name='runtime-memory')
  # Determine the number of new frames in the input, such that we only operate
  # on those. For training we do not use the memory, and thus use all frames
  # provided in the input.
  # new_fingerprint_input: [batch, num_new_frames*input_frequency_size]
  if is_training:
    num_new_frames = input_time_size
  else:
    window_stride_ms = int(model_settings['window_stride_samples'] * 1000 /
                           model_settings['sample_rate'])
    num_new_frames = tf.cond(
        tf.equal(tf.count_nonzero(memory), 0),
        lambda: input_time_size,
        lambda: int(runtime_settings['clip_stride_ms'] / window_stride_ms))
  new_fingerprint_input = fingerprint_input[
      :, -num_new_frames*input_frequency_size:]
  # Expand to add input channels dimension.
  new_fingerprint_input = tf.expand_dims(new_fingerprint_input, 2)

  # Create the frequency filters.
  weights_frequency = tf.Variable(
      tf.truncated_normal([input_frequency_size, num_filters], stddev=0.01))
  # Expand to add input channels dimensions.
  # weights_frequency: [input_frequency_size, 1, num_filters]
  weights_frequency = tf.expand_dims(weights_frequency, 1)
  # Convolve the 1D feature filters sliding over the time dimension.
  # activations_time: [batch, num_new_frames, num_filters]
  activations_time = tf.nn.conv1d(
      new_fingerprint_input, weights_frequency, input_frequency_size, 'VALID')
  # Rearrange such that we can perform the batched matmul.
  # activations_time: [num_filters, batch, num_new_frames]
  activations_time = tf.transpose(activations_time, perm=[2, 0, 1])

  # Runtime memory optimization.
  if not is_training:
    # We need to drop the activations corresponding to the oldest frames, and
    # then add those corresponding to the new frames.
    new_memory = memory[:, :, num_new_frames:]
    new_memory = tf.concat([new_memory, activations_time], 2)
    tf.assign(memory, new_memory)
    activations_time = new_memory

  # Create the time filters.
  weights_time = tf.Variable(
      tf.truncated_normal([num_filters, input_time_size], stddev=0.01))
  # Apply the time filter on the outputs of the feature filters.
  # weights_time: [num_filters, input_time_size, 1]
  # outputs: [num_filters, batch, 1]
  weights_time = tf.expand_dims(weights_time, 2)
  outputs = tf.matmul(activations_time, weights_time)
  # Split num_units and rank into separate dimensions (the remaining
  # dimension is the input_shape[0] -i.e. batch size). This also squeezes
  # the last dimension, since it's not used.
  # [num_filters, batch, 1] => [num_units, rank, batch]
  outputs = tf.reshape(outputs, [num_units, rank, -1])
  # Sum the rank outputs per unit => [num_units, batch].
  units_output = tf.reduce_sum(outputs, axis=1)
  # Transpose to shape [batch, num_units]
  units_output = tf.transpose(units_output)

  # Appy bias.
  bias = tf.Variable(tf.zeros([num_units]))
  first_bias = tf.nn.bias_add(units_output, bias)

  # Relu.
  first_relu = tf.nn.relu(first_bias)

  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu

  first_fc_output_channels = 256
  first_fc_weights = tf.Variable(
      tf.truncated_normal([num_units, first_fc_output_channels], stddev=0.01))
  first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
  first_fc = tf.matmul(first_dropout, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 256
  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
  second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_fc_output_channels, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc

def create_dnn_model(fingerprint_input, model_settings, model_size_info, 
                       is_training):
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
  tf.summary.histogram('input', flow)
  for i in range(1, num_layers + 1):
      with tf.variable_scope('fc'+str(i)):
          W = tf.get_variable('W', shape=[layer_dim[i-1], layer_dim[i]], 
                initializer=tf.contrib.layers.xavier_initializer())
          tf.summary.histogram('fc_'+str(i)+'_w', W)
          b = tf.get_variable('b', shape=[layer_dim[i]])
          tf.summary.histogram('fc_'+str(i)+'_b', b)
          flow = tf.matmul(flow, W) + b
          flow = tf.nn.relu(flow)
          if is_training:
            flow = tf.nn.dropout(flow, dropout_prob)

  weights = tf.get_variable('final_fc', shape=[layer_dim[-1], label_count], 
              initializer=tf.contrib.layers.xavier_initializer())
  bias = tf.Variable(tf.zeros([label_count]))
  logits = tf.matmul(flow, weights) + bias
  if is_training:
    return logits, dropout_prob
  else:
    return logits

def create_cnn_model(fingerprint_input, model_settings, model_size_info,
                       is_training):
  """Builds a model with 2 convolution layers followed by a linear layer and 
      a hidden fully-connected layer.
  model_size_info: defines the first and second convolution parameters in
      {number of conv features, conv filter height, width, stride in y,x dir.},
      followed by linear layer size and fully-connected layer size.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])

  first_filter_count = model_size_info[0] 
  first_filter_height = model_size_info[1]   #time axis
  first_filter_width = model_size_info[2]    #frequency axis
  first_filter_stride_y = model_size_info[3] #time axis
  first_filter_stride_x = model_size_info[4] #frequency_axis

  second_filter_count = model_size_info[5] 
  second_filter_height = model_size_info[6]   #time axis
  second_filter_width = model_size_info[7]    #frequency axis
  second_filter_stride_y = model_size_info[8] #time axis
  second_filter_stride_x = model_size_info[9] #frequency_axis
 
  linear_layer_size = model_size_info[10]
  fc_size = model_size_info[11]

  # first conv
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
      1, first_filter_stride_y, first_filter_stride_x, 1
  ], 'VALID') + first_bias
  first_conv = tf.layers.batch_normalization(first_conv, training=is_training,
                 name='bn1')
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  first_conv_output_width = math.ceil(
      (input_frequency_size - first_filter_width + 1) /
      first_filter_stride_x)
  first_conv_output_height = math.ceil(
      (input_time_size - first_filter_height + 1) /
      first_filter_stride_y)

  # second conv
  second_weights = tf.Variable(
      tf.truncated_normal(
          [second_filter_height, second_filter_width, first_filter_count, 
             second_filter_count],
          stddev=0.01))
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(first_dropout, second_weights, [
      1, second_filter_stride_y, second_filter_stride_x, 1
  ], 'VALID') + second_bias
  second_conv = tf.layers.batch_normalization(second_conv, training=is_training,
                  name='bn2')
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_output_width = math.ceil(
      (first_conv_output_width - second_filter_width + 1) /
      second_filter_stride_x)
  second_conv_output_height = math.ceil(
      (first_conv_output_height - second_filter_height + 1) /
      second_filter_stride_y)

  second_conv_element_count = int(
      second_conv_output_width*second_conv_output_height*second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout,
                                    [-1, second_conv_element_count])

  # linear layer
  W = tf.get_variable('W', shape=[second_conv_element_count, linear_layer_size],
        initializer=tf.contrib.layers.xavier_initializer())
  b = tf.get_variable('b', shape=[linear_layer_size])
  flow = tf.matmul(flattened_second_conv, W) + b

  # first fc
  first_fc_output_channels = fc_size
  first_fc_weights = tf.Variable(
      tf.truncated_normal(
          [linear_layer_size, first_fc_output_channels], stddev=0.01))
  first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
  first_fc = tf.matmul(flow, first_fc_weights) + first_fc_bias
  first_fc = tf.layers.batch_normalization(first_fc, training=is_training, 
               name='bn3')
  first_fc = tf.nn.relu(first_fc)
  if is_training:
    final_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    final_fc_input = first_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_fc_output_channels, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_basic_lstm_model(fingerprint_input, model_settings, model_size_info, 
                              is_training):
  """Builds a model with a basic lstm layer (without output projection and 
       peep-hole connections)
  model_size_info: defines the number of memory cells in basic lstm model
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size])

  num_classes = model_settings['label_count']

  if type(model_size_info) is list:
    LSTM_units = model_size_info[0]
  else:
    LSTM_units = model_size_info 

  with tf.name_scope('LSTM-Layer'):
    with tf.variable_scope("lstm"): 
      lstmcell = tf.contrib.rnn.BasicLSTMCell(LSTM_units, forget_bias=1.0, 
                   state_is_tuple=True)
      _, last = tf.nn.dynamic_rnn(cell=lstmcell, inputs=fingerprint_4d, 
                  dtype=tf.float32)
      flow = last[-1]

  with tf.name_scope('Output-Layer'):
    W_o = tf.get_variable('W_o', shape=[LSTM_units, num_classes], 
            initializer=tf.contrib.layers.xavier_initializer())
    b_o = tf.get_variable('b_o', shape=[num_classes])
    logits = tf.matmul(flow, W_o) + b_o

  if is_training:
    return logits, dropout_prob
  else:
    return logits

def create_lstm_model(fingerprint_input, model_settings, model_size_info, 
                        is_training):
  """Builds a model with a lstm layer (with output projection layer and 
       peep-hole connections)
  Based on model described in https://arxiv.org/abs/1705.02411
  model_size_info: [projection size, memory cells in LSTM]
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size])

  num_classes = model_settings['label_count']
  projection_units = model_size_info[0]
  LSTM_units = model_size_info[1]
  with tf.name_scope('LSTM-Layer'):
    with tf.variable_scope("lstm"): 
      lstmcell = tf.contrib.rnn.LSTMCell(LSTM_units, use_peepholes=True, 
                   num_proj=projection_units)
      _, last = tf.nn.dynamic_rnn(cell=lstmcell, inputs=fingerprint_4d, 
                  dtype=tf.float32)
      flow = last[-1]

  with tf.name_scope('Output-Layer'):
    W_o = tf.get_variable('W_o', shape=[projection_units, num_classes], 
            initializer=tf.contrib.layers.xavier_initializer())
    b_o = tf.get_variable('b_o', shape=[num_classes])
    logits = tf.matmul(flow, W_o) + b_o

  if is_training:
    return logits, dropout_prob
  else:
    return logits

class LayerNormGRUCell(rnn_cell_impl.RNNCell):

  def __init__(self, num_units, forget_bias=1.0,
               input_size=None, activation=math_ops.tanh,
               layer_norm=True, norm_gain=1.0, norm_shift=0.0,
               dropout_keep_prob=1.0, dropout_prob_seed=None,
               reuse=None):

    super(LayerNormGRUCell, self).__init__(_reuse=reuse)

    if input_size is not None:
      tf.logging.info("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._g = norm_gain
    self._b = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(self._g)
    beta_init = init_ops.constant_initializer(self._b)
    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init)
      vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def _linear(self, args, copy):
    out_size = copy * self._num_units
    proj_size = args.get_shape()[-1]
    weights = vs.get_variable("kernel", [proj_size, out_size])
    out = math_ops.matmul(args, weights)
    if not self._layer_norm:
      bias = vs.get_variable("bias", [out_size])
      out = nn_ops.bias_add(out, bias)
    return out

  def call(self, inputs, state):
    """LSTM cell with layer normalization and recurrent dropout."""
    with vs.variable_scope("gates"):
      h = state
      args = array_ops.concat([inputs, h], 1)
      concat = self._linear(args, 2)

      z, r = array_ops.split(value=concat, num_or_size_splits=2, axis=1)
      if self._layer_norm:
        z = self._norm(z, "update") 
        r = self._norm(r, "reset")

    with vs.variable_scope("candidate"):
      args = array_ops.concat([inputs, math_ops.sigmoid(r) * h], 1)
      new_c = self._linear(args, 1)
      if self._layer_norm:
        new_c = self._norm(new_c, "state")
    new_h = self._activation(new_c) * math_ops.sigmoid(z) + \
              (1 - math_ops.sigmoid(z)) * h
    return new_h, new_h

def create_gru_model(fingerprint_input, model_settings, model_size_info, 
                       is_training):
  """Builds a model with multi-layer GRUs
  model_size_info: [number of GRU layers, number of GRU cells per layer]
  Optionally, the bi-directional GRUs and/or GRU with layer-normalization 
    can be explored.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size])

  num_classes = model_settings['label_count']

  layer_norm = False
  bidirectional = False

  num_layers = model_size_info[0]
  gru_units = model_size_info[1]

  gru_cell_fw = []
  gru_cell_bw = []
  if layer_norm:
    for i in range(num_layers):
      gru_cell_fw.append(LayerNormGRUCell(gru_units))
      if bidirectional:
        gru_cell_bw.append(LayerNormGRUCell(gru_units))
  else:
    for i in range(num_layers):
      gru_cell_fw.append(tf.contrib.rnn.GRUCell(gru_units))
      if bidirectional:
        gru_cell_bw.append(tf.contrib.rnn.GRUCell(gru_units))
  
  if bidirectional:
    outputs, output_state_fw, output_state_bw = \
      tf.contrib.rnn.stack_bidirectional_dynamic_rnn(gru_cell_fw, gru_cell_bw, 
      fingerprint_4d, dtype=tf.float32)
    flow = outputs[:, -1, :]
  else:
    cells = tf.contrib.rnn.MultiRNNCell(gru_cell_fw)
    _, last = tf.nn.dynamic_rnn(cell=cells, inputs=fingerprint_4d, 
                dtype=tf.float32)
    flow = last[-1]

  with tf.name_scope('Output-Layer'):
    W_o = tf.get_variable('W_o', shape=[flow.get_shape()[-1], num_classes], 
            initializer=tf.contrib.layers.xavier_initializer())
    b_o = tf.get_variable('b_o', shape=[num_classes])
    logits = tf.matmul(flow, W_o) + b_o

  if is_training:
    return logits, dropout_prob
  else:
    return logits
  

def create_crnn_model(fingerprint_input, model_settings,
                                  model_size_info, is_training):
  """Builds a model with convolutional recurrent networks with GRUs
  Based on the model definition in https://arxiv.org/abs/1703.05390
  model_size_info: defines the following convolution layer parameters
      {number of conv features, conv filter height, width, stride in y,x dir.},
      followed by number of GRU layers and number of GRU cells per layer
  Optionally, the bi-directional GRUs and/or GRU with layer-normalization 
    can be explored.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])

  layer_norm = False
  bidirectional = False

  # CNN part
  first_filter_count = model_size_info[0]
  first_filter_height = model_size_info[1]
  first_filter_width = model_size_info[2]
  first_filter_stride_y = model_size_info[3]
  first_filter_stride_x = model_size_info[4]

  first_weights = tf.get_variable('W', shape=[first_filter_height, 
                    first_filter_width, 1, first_filter_count], 
    initializer=tf.contrib.layers.xavier_initializer())

  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
      1, first_filter_stride_y, first_filter_stride_x, 1
  ], 'VALID') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  first_conv_output_width = int(math.floor(
      (input_frequency_size - first_filter_width + first_filter_stride_x) /
      first_filter_stride_x))
  first_conv_output_height = int(math.floor(
      (input_time_size - first_filter_height + first_filter_stride_y) /
      first_filter_stride_y))

  # GRU part
  num_rnn_layers = model_size_info[5]
  RNN_units = model_size_info[6]
  flow = tf.reshape(first_dropout, [-1, first_conv_output_height, 
           first_conv_output_width * first_filter_count])
  cell_fw = []
  cell_bw = []
  if layer_norm:
    for i in range(num_rnn_layers):
      cell_fw.append(LayerNormGRUCell(RNN_units))
      if bidirectional:
        cell_bw.append(LayerNormGRUCell(RNN_units))
  else:
    for i in range(num_rnn_layers):
      cell_fw.append(tf.contrib.rnn.GRUCell(RNN_units))
      if bidirectional:
        cell_bw.append(tf.contrib.rnn.GRUCell(RNN_units))

  if bidirectional:
    outputs, output_state_fw, output_state_bw = \
      tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, flow, 
      dtype=tf.float32)
    flow_dim = first_conv_output_height*RNN_units*2
    flow = tf.reshape(outputs, [-1, flow_dim])
  else:
    cells = tf.contrib.rnn.MultiRNNCell(cell_fw)
    _, last = tf.nn.dynamic_rnn(cell=cells, inputs=flow, dtype=tf.float32)
    flow_dim = RNN_units
    flow = last[-1]

  first_fc_output_channels = model_size_info[7]

  first_fc_weights = tf.get_variable('fcw', shape=[flow_dim, 
    first_fc_output_channels], 
    initializer=tf.contrib.layers.xavier_initializer())
  
  first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
  first_fc = tf.nn.relu(tf.matmul(flow, first_fc_weights) + first_fc_bias)
  if is_training:
    final_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    final_fc_input = first_fc

  label_count = model_settings['label_count']
  
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_fc_output_channels, label_count], stddev=0.01))
  
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc

def create_ds_cnn_model(fingerprint_input, model_settings, model_size_info, 
                          is_training):
  """Builds a model with depthwise separable convolutional neural network
  Model definition is based on https://arxiv.org/abs/1704.04861 and
  Tensorflow implementation: https://github.com/Zehaos/MobileNet

  model_size_info: defines number of layers, followed by the DS-Conv layer
    parameters in the order {number of conv features, conv filter height, 
    width and stride in y,x dir.} for each of the layers. 
  Note that first layer is always regular convolution, but the remaining 
    layers are all depthwise separable convolutions.
  """

  def ds_cnn_arg_scope(weight_decay=0):
    """Defines the default ds_cnn argument scope.
    Args:
      weight_decay: The weight decay to use for regularizing the model.
    Returns:
      An `arg_scope` to use for the DS-CNN model.
    """
    with slim.arg_scope(
        [slim.convolution2d, slim.separable_convolution2d],
        weights_initializer=slim.initializers.xavier_initializer(),
        biases_initializer=slim.init_ops.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
      return sc

  def _depthwise_separable_conv(inputs,
                                num_pwc_filters,
                                sc,
                                kernel_size,
                                stride):
    """ Helper function to build the depth-wise separable convolution layer.
    """

    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  depth_multiplier=1,
                                                  kernel_size=kernel_size,
                                                  scope=sc+'/dw_conv')

    bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_conv/batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pw_conv')
    bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_conv/batch_norm')
    return bn


  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  label_count = model_settings['label_count']
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
 
  t_dim = input_time_size
  f_dim = input_frequency_size

  #Extract model dimensions from model_size_info
  num_layers = model_size_info[0]
  conv_feat = [None]*num_layers
  conv_kt = [None]*num_layers
  conv_kf = [None]*num_layers
  conv_st = [None]*num_layers
  conv_sf = [None]*num_layers
  i=1
  for layer_no in range(0,num_layers):
    conv_feat[layer_no] = model_size_info[i]
    i += 1
    conv_kt[layer_no] = model_size_info[i]
    i += 1
    conv_kf[layer_no] = model_size_info[i]
    i += 1
    conv_st[layer_no] = model_size_info[i]
    i += 1
    conv_sf[layer_no] = model_size_info[i]
    i += 1

  scope = 'DS-CNN'
  with tf.variable_scope(scope) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        activation_fn=None,
                        weights_initializer=slim.initializers.xavier_initializer(),
                        biases_initializer=slim.init_ops.zeros_initializer(),
                        outputs_collections=[end_points_collection]):
      with slim.arg_scope([slim.batch_norm],
                          is_training=is_training,
                          decay=0.96,
                          updates_collections=None,
                          activation_fn=tf.nn.relu):
        for layer_no in range(0,num_layers):
          if layer_no==0:
            net = slim.convolution2d(fingerprint_4d, conv_feat[layer_no],\
                      [conv_kt[layer_no], conv_kf[layer_no]], stride=[conv_st[layer_no], conv_sf[layer_no]], padding='SAME', scope='conv_1')
            net = slim.batch_norm(net, scope='conv_1/batch_norm')
          else:
            net = _depthwise_separable_conv(net, conv_feat[layer_no], \
                      kernel_size = [conv_kt[layer_no],conv_kf[layer_no]], \
                      stride = [conv_st[layer_no],conv_sf[layer_no]], sc='conv_ds_'+str(layer_no))
          t_dim = math.ceil(t_dim/float(conv_st[layer_no]))
          f_dim = math.ceil(f_dim/float(conv_sf[layer_no]))

        net = slim.avg_pool2d(net, [t_dim, f_dim], scope='avg_pool')

    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
    logits = slim.fully_connected(net, label_count, activation_fn=None, scope='fc1')

  if is_training:
    return logits, dropout_prob
  else:
    return logits

