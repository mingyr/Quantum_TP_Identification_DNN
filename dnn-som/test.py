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
import numpy as np
import tensorflow as tf
import sonnet as snt

from config import FLAGS
from input_ import Input
from model import Model
from scipy import io
import matlab.engine

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.ERROR)

    # tf.logging.set_verbosity(3)  # Print INFO log messages.

    summaries = []
    assert FLAGS.test_size > 0, 'batch size for testing'

    if FLAGS.data_dir == '' or not os.path.exists(FLAGS.data_dir):
        raise ValueError('invalid data directory {}'.format(FLAGS.data_dir))

    test_data_path = os.path.join(FLAGS.data_dir, 'quantum-test.tfr')

    if FLAGS.output_dir == '':
        raise ValueError('invalid output directory {}'.format(FLAGS.output_dir))

    checkpoint_dir = os.path.join(FLAGS.output_dir, '')

    test_input = Input(1, [FLAGS.img_height, FLAGS.img_width, FLAGS.num_channels])

    inputs, labels = test_input(test_data_path)

    model = Model(act = FLAGS.activation, pool = FLAGS.pooling, with_memory = FLAGS.with_memory, log = False)

    inputs, labels = test_input(test_data_path)

    logits = model(inputs, training = False)

    logit_indices = tf.argmax(logits, axis = -1)

    # Define the metric and update operations
    metric_op, metric_update_op = tf.metrics.accuracy(labels, logit_indices)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        # sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
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

        logits_lst = []
        labels_lst = []
        indices_lst = []

        for i in range(FLAGS.test_size):
            _, logits_val, indices_val, labels_val = sess.run([metric_update_op, logits, logit_indices, labels])
            logits_lst.append(logits_val)
            labels_lst.append(labels_val)
            indices_lst.append(indices_val)

        accu = sess.run(metric_op)
        print("accu -> {}".format(accu))

        io.savemat("tsne.mat", {"logits": logits_lst, 'classes': labels_lst, 'indices': indices_lst})

        eng = matlab.engine.start_matlab("-nodisplay -nosplash")
        accus = eng.calc_class_accu(matlab.int32(np.squeeze(labels_lst).tolist()), 
                                    matlab.int32(np.squeeze(indices_lst).tolist()))
        eng.quit()

        for i, a in enumerate(np.squeeze(np.array(accus)).tolist()):
            print("C = {}: {}".format(i - 2, a))


if __name__ == "__main__":
    tf.app.run()
