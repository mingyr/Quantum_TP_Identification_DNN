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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import sonnet as snt

from config import FLAGS
from input_3 import Input
from model import Model
from utils import TestClassification
from scipy import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.ERROR)

    # tf.logging.set_verbosity(3)  # Print INFO log messages.

    summaries = []
    assert FLAGS.test_size > 0, 'batch size for testing'                                                        

    if FLAGS.data_dir == '' or not os.path.isfile(FLAGS.data_dir):
        raise ValueError('invalid file name {}'.format(FLAGS.data_dir))

    if FLAGS.output_dir == '':
        raise ValueError('invalid output directory {}'.format(FLAGS.output_dir))

    checkpoint_dir = os.path.join(FLAGS.output_dir, '')

    test_input = Input(1, [FLAGS.img_height, FLAGS.img_width, FLAGS.num_channels])

    inputs, coords = test_input(FLAGS.data_dir)

    model = Model(act = FLAGS.activation, pool = FLAGS.pooling, log = False)

    logits = model(inputs, training = False)
    logits = tf.nn.softmax(tf.squeeze(logits))

    indices = tf.argmax(logits, axis = -1)

    # we check the confidence
    confidence = tf.constant(FLAGS.confidence, tf.float32, [FLAGS.num_classes])
    with tf.control_dependencies([tf.assert_equal(tf.shape(logits), tf.shape(confidence))]):
        candidates = tf.math.greater(logits, confidence)

    indices = tf.cond(tf.reduce_any(candidates), lambda: indices, lambda: tf.constant(FLAGS.num_classes, tf.int64))

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

        indices_lst = []
        coords_lst = []

        for i in range(FLAGS.test_size):
            indices_val, coords_val = sess.run([indices, coords])
            indices_lst.append(indices_val)
            coords_lst.append(coords_val)

        comp1 = FLAGS.data_dir.split('/')[-2]
        comp2 = FLAGS.output_dir.split('/')[-1]
        filename = "-".join((comp1, comp2, "indices-coords-{}.mat".format(FLAGS.confidence)))

        io.savemat(filename, {"indices": indices_lst, 'coords': coords_lst})

if __name__ == "__main__":
    tf.app.run()
