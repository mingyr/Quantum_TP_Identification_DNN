# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DNC configuration file"""

import tensorflow as tf

# General parameters
tf.flags.DEFINE_integer('num_gpus', 2, 'number of GPUs for training.')
tf.flags.DEFINE_string('gpus', '', 'visible GPU list')
tf.flags.DEFINE_string('type', '', 'type of the model, classification or regression')

# Optimizer parameters
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10, "Epsilon used for RMSProp optimizer.")

# Input data general parameters
tf.flags.DEFINE_string("data_type", "", "momentum or position")
tf.flags.DEFINE_integer('img_height', 599, 'height of the spectrum image.')
tf.flags.DEFINE_integer('img_width', 599, 'width of the spectrum image.')
tf.flags.DEFINE_integer('img_size', 599, 'size of the spectrum image.')
tf.flags.DEFINE_integer('num_classes', 5, 'number of classes to classify.')

# begin of momentum data specification
'''
tf.flags.DEFINE_integer('num_channels', 4, 'number of channels of spectrum image')
tf.flags.DEFINE_float("learning_rate", 0.001, "Optimizer learning rate.")
tf.flags.DEFINE_integer("lr_decay_steps", 100, "Number of steps for learning rate decay.")
tf.flags.DEFINE_float("lr_decay_factor", 0.80, "Learning rate decay factor.")

tf.flags.DEFINE_float("lr_som", 0.40, "SOM learning rate.")
'''
# end of momentum data specification

# begin of position data specification

tf.flags.DEFINE_integer('num_channels', 2, 'number of channels of position image')
tf.flags.DEFINE_float("learning_rate", 0.001, "Optimizer learning rate.")
tf.flags.DEFINE_integer("lr_decay_steps", 200, "Number of steps for learning rate decay.")
tf.flags.DEFINE_float("lr_decay_factor", 0.90, "Learning rate decay factor.")

tf.flags.DEFINE_float("lr_som", 0.4, "SOM1 learning rate.")

# end of position data specification

# Model parameters
# Memory part parameters
tf.flags.DEFINE_boolean("with_memory", True, "simplified postion label or not")
tf.flags.DEFINE_integer("memory_height", 256, "The height of memory region 1.")
tf.flags.DEFINE_integer("memory_width", 256, "The width of memory region 1.")

# Computation part parameters
tf.flags.DEFINE_integer('num_outputs_block_1', 8, 'number of outputs of each layers of block 1.')
tf.flags.DEFINE_integer('num_outputs_block_2', 8, 'number of outputs of each layers of block 2.')
tf.flags.DEFINE_integer('num_outputs_block_3', 16, 'number of outputs of each layers of block 3.')
tf.flags.DEFINE_integer('num_outputs_block_4', 16, 'number of outputs of each layers of block 4.')
tf.flags.DEFINE_integer('num_outputs_block_5', 32, 'number of outputs of each layers of block 5.')
tf.flags.DEFINE_integer('num_outputs_block_6', 32, 'number of outputs of each layers of block 6.')

tf.flags.DEFINE_integer('num_outputs_dense', 512, 'number of nodes of dense layer.')
tf.flags.DEFINE_integer('filter_size', 5, 'filter size of conv layers.')
tf.flags.DEFINE_string('activation', 'elu', 'type of activation function')
tf.flags.DEFINE_string('pooling', 'max', 'type of pooling')
tf.flags.DEFINE_boolean("residual", True, "whether or not to utilize residual structure")

# Training options
tf.flags.DEFINE_integer("num_iterations", 1000, "Number of iterations to train for.")
tf.flags.DEFINE_integer('train_batch_size', 0, 'batch size for training.')
tf.flags.DEFINE_integer('xval_batch_size', 0, 'batch size for cross evaluation')

# Test options
# for normal case, 736
# for transitional data, 937
tf.flags.DEFINE_integer('test_size', 0, 'sample size for test')

# Directory options
tf.flags.DEFINE_string('data_dir', '', 'directory for model checkpoints.')
tf.flags.DEFINE_string('output_dir', '', 'directory for model checkpoints.')
tf.flags.DEFINE_integer("report_interval", 5, "Iterations between reports (samples, valid loss).")

# Miscellaneous options
tf.flags.DEFINE_boolean("validate", True, "whether or not do validation")

tf.flags.DEFINE_float("confidence", 0.0, "confidence")

FLAGS = tf.flags.FLAGS


