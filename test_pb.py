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
# Modifications Copyright 2017-18 Arm Inc. All Rights Reserved. 
# Adapted from freeze.py to run inference on train/val/test dataset on the 
# trained model in the form of pb
#          
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import tensorflow as tf
import input_data
import models


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

def run_inference_pb(wanted_words, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, dct_coefficient_count, 
                           model_architecture, model_size_info):
  """Creates an audio model with the nodes needed for inference.

  Uses the supplied arguments to create a model, and inserts the input and
  output nodes that are needed to use the graph for inference.

  Args:
    wanted_words: Comma-separated list of the words we're trying to recognize.
    sample_rate: How many samples per second are in the input audio files.
    clip_duration_ms: How many samples to analyze for the audio pattern.
    window_size_ms: Time slice duration to estimate frequencies from.
    window_stride_ms: How far apart time slices should be.
    dct_coefficient_count: Number of frequency bands to analyze.
    model_architecture: Name of the kind of model to generate.
    model_size_info: Model dimensions : different lengths for different models
  """
  
  tf.logging.set_verbosity(tf.logging.INFO)
  sess = tf.InteractiveSession()
  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, dct_coefficient_count)

  load_graph(FLAGS.graph)

  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings)

  softmax = sess.graph.get_tensor_by_name("labels_softmax:0")

  label_count = model_settings['label_count']

  ground_truth_input = tf.placeholder(
      tf.float32, [None, label_count], name='groundtruth_input')

  predicted_indices = tf.argmax(softmax, 1)
  expected_indices = tf.argmax(ground_truth_input, 1)
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  confusion_matrix = tf.confusion_matrix(
      expected_indices, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  if(FLAGS.training):
    # training set
    set_size = audio_processor.set_size('training')
    tf.logging.info('Training set size:%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None
    for i in range(0, set_size):
      training_file, training_ground_truth = (
          audio_processor.get_wav_files(1, i, model_settings, 'training'))
      with open(training_file[0], 'rb') as wav_file:
        training_data = wav_file.read()
      training_accuracy, conf_matrix = sess.run(
          [evaluation_step, confusion_matrix],
          feed_dict={
              'wav_data:0': training_data,
              ground_truth_input: training_ground_truth,
          })
      total_accuracy += (training_accuracy) / set_size
      if total_conf_matrix is None:
        total_conf_matrix = conf_matrix
      else:
        total_conf_matrix += conf_matrix
    tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.logging.info('Training accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                             set_size))

  # validation set
  set_size = audio_processor.set_size('validation')
  tf.logging.info('Validation set size:%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in range(0, set_size):
    validation_file, validation_ground_truth = (
        audio_processor.get_wav_files(1, i, model_settings, 'validation'))
    with open(validation_file[0], 'rb') as wav_file:
      validation_data = wav_file.read()
    validation_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            'wav_data:0': validation_data,
            ground_truth_input: validation_ground_truth,
        })
    total_accuracy += (validation_accuracy) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Validation accuracy = %.2f%% (N=%d)' %
                  (total_accuracy * 100, set_size))
  # test set
  set_size = audio_processor.set_size('testing')
  tf.logging.info('Test set size:%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in range(0, set_size):
    test_file, test_ground_truth = (
        audio_processor.get_wav_files(1, i, model_settings, 'testing'))
    with open(test_file[0], 'rb') as wav_file:
      test_data = wav_file.read()
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            'wav_data:0': test_data,
            ground_truth_input: test_ground_truth,
        })
    total_accuracy += (test_accuracy) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                           set_size))

def main(_):

  # Create the model, load weights from pb and run on train/val/test
  run_inference_pb(FLAGS.wanted_words, FLAGS.sample_rate,
      FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count,
      FLAGS.model_architecture, FLAGS.model_size_info)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='dnn',
      help='What model architecture to use')
  parser.add_argument(
      '--model_size_info',
      type=int,
      nargs="+",
      default=[128,128,128],
      help='Model dimensions - different for various models')
  parser.add_argument(
      '--training',
      action='store_true',
      default=False,
      help='run on training (1) or not (0)',)

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
