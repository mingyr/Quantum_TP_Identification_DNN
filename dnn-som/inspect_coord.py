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
"""Example script to train the DNC on a repeated copy task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import sonnet as snt

from config import FLAGS
from input_2 import Input
from model import Model
from utils import Metrics, reset_metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.ERROR)

    # tf.logging.set_verbosity(3)  # Print INFO log messages.

    if FLAGS.data_dir == '':
        raise ValueError('invalid file name {}'.format(FLAGS.data_dir))
    else:
        data_filenames = FLAGS.data_dir.split(",")
        exists = [os.path.isfile(filename) for filename in data_filenames]
        indices = [i for i, b in enumerate(exists) if not b]
        if len(indices) > 0:
            raise ValueError('invalid file name {}'.format(data_filenames[indices[0]]))

    checkpoint_dir = os.path.join(FLAGS.output_dir, '')

    test_input = Input(1, [FLAGS.img_height, FLAGS.img_width, FLAGS.num_channels])

    filename_tensor = tf.placeholder(tf.string, shape=[None]) 
    inputs, labels = test_input(filename_tensor)

    model = Model(act = FLAGS.activation, pool = FLAGS.pooling, with_memory = FLAGS.with_memory, log = True)

    logits = model(inputs, training = False)

    logit_indices = tf.argmax(logits, axis = -1)

    # Define the metric and update operations
    metrics = Metrics("accuracy")
    metric_op, metric_update_op = metrics(labels, logit_indices)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        sess.run([tf.local_variables_initializer()])

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint.
            saver.restore(sess, ckpt.model_checkpoint_path)

            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/imagenet_train/model.ckpt-0,
            # extract global_step from it.
            # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            # print('Successfully loaded model from %s at step = %s.' %
            #       (ckpt.model_checkpoint_path, global_step))

            print('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)

        else:
            print('No checkpoint file found')
            return

        for filename in data_filenames:    
            sess.run([metric_update_op, logits], feed_dict = {filename_tensor: [filename]})

            accu = sess.run(metric_op)
            print("accu -> {}".format(accu))

            reset_metrics(sess)

if __name__ == "__main__":
    tf.app.run()

