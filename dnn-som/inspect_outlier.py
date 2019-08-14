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
import numpy as np
import tensorflow as tf
import sonnet as snt

from config import FLAGS
from input_ import Input
from model import Model
from scipy import io
import pickle
import warnings

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

    if FLAGS.data_type == "momentum":
        filenames_datafile = '/scratch/quantum-meta/filenames_transition_momentum.dat'
        warnings.warn("Check the data file containing file names: {}".format(filenames_datafile), UserWarning) 
        with open(filenames_datafile, 'rb') as f:
            filenames = pickle.load(f)
    elif FLAGS.data_type == "position":
        filenames_datafile = '/scratch/quantum-meta/filenames_transition_position.dat'
        warnings.warn("Check the data file containing file names: {}".format(filenames_datafile), UserWarning)        
        with open(filenames_datafile, 'rb') as f:
            filenames = pickle.load(f)
    else:
        print("invalid data type {}".format(FLAGS.data_type))
        return 

    test_filenames = filenames['test_filenames']

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

        image_data_placeholder = tf.placeholder(tf.float32, [FLAGS.img_height, FLAGS.img_width, FLAGS.num_channels])
        image_descr_placeholder = tf.placeholder(tf.string, [], name = "image_descr")

        image_data = tf.unstack(image_data_placeholder, axis = -1)
        
        image_data = [(data - tf.reshape(tf.reduce_min(tf.reduce_min(data)), [1, 1])) / \
                      tf.reshape((tf.reduce_max(tf.reduce_max(data)) -tf.reduce_min(tf.reduce_min(data))), [1, 1]) \
                      for data in image_data]

        image_data = [tf.expand_dims(data, axis = -1) for data in image_data]
        image_data = [tf.expand_dims(data, axis = 0) for data in image_data]

        add_img_op = [tf.summary.image("misclassified image data {}th component".format(i), image_data[i]) 
                      for i in range(FLAGS.num_channels)]
        add_txt_op = [tf.summary.text("misclassified image description {}th component".format(i), image_descr_placeholder) 
                      for i in range(FLAGS.num_channels)]
 
        add_summ = tf.summary.merge_all()

        writer = tf.summary.FileWriter("output-outlier", tf.get_default_graph())

        logit_indices = tf.squeeze(logit_indices)
        labels = tf.squeeze(labels)

        for i in range(FLAGS.test_size):
            _, images_val, logits_val, indices_val, labels_val = sess.run([metric_update_op, inputs, logits, logit_indices, labels])
            logits_lst.append(logits_val)
            labels_lst.append(labels_val)
            indices_lst.append(indices_val)

            # print("{}: ground-truth label: {}, predicted label: {}".format(i, labels_val, indices_val))

            if indices_val != labels_val:
                print("{}: ground-truth {}, predicted {}".format(i, labels_val, indices_val))

                filename = test_filenames[i]
                filename = filename.split('\\')[-1]
                print("'{}'".format(filename))

                summ_str = sess.run(add_summ, feed_dict = {image_data_placeholder: np.squeeze(images_val, axis = 0), 
                                                           image_descr_placeholder: "{}: ground-truth {}, predicted {}".format(filename, labels_val, indices_val)})
                writer.add_summary(summ_str, i)
 
        accu = sess.run(metric_op)
        print("accu -> {}".format(accu))

        # io.savemat("logits.mat", {"logits": logits_lst, 'classes': labels_lst, 'indices': indices_lst})

        writer.close()

if __name__ == "__main__":
    tf.app.run()
