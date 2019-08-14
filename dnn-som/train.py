# ==============================================================================
# This script is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Training routine"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import sonnet as snt

from config import FLAGS
from input_ import Input
from model import Model 
from utils import LossClassification, XValClassification, Summaries
from optimizer import RMSProp, Adam


def train(model, data_path, cur_iter, summ):
    input_ = Input(FLAGS.train_batch_size, [FLAGS.img_height, FLAGS.img_width, FLAGS.num_channels])

    images, labels = input_(data_path)
    logits = model(images, cur_iter = cur_iter)

    # Calculate the loss of the model.
    loss = LossClassification(FLAGS.num_classes, gpu = True)(logits, labels)

    # Create an optimizer that performs gradient descent.
    optimizer = Adam(FLAGS.learning_rate, lr_decay = False, 
                     lr_decay_steps = FLAGS.lr_decay_steps,
                     lr_decay_factor = FLAGS.lr_decay_factor)

    train_op = optimizer(loss)

    summ.register('train', 'train_loss', loss)
    summ.register('train', 'learning_rate', optimizer.lr)

    train_summ_op = summ('train')

    return loss, train_op, train_summ_op

def xval(model, data_path, summ):

    input_ = Input(FLAGS.xval_batch_size, [FLAGS.img_height, FLAGS.img_width, FLAGS.num_channels])

    images, labels, = input_(data_path)
    logits = model(images, training = False)

    xval_accu_op = XValClassification(gpu = True)(logits, labels)

    summ.register('xval', 'xval_accuracy', xval_accu_op)
    xval_summ_op = summ('xval')

    return xval_accu_op, xval_summ_op

def main(unused_argv):
    tf.logging.set_verbosity(3)  # Print INFO log messages.

    """Trains the model and periodically reports the loss."""

    if FLAGS.data_type == 'momentum':
        assert (FLAGS.num_channels == 4)
    elif FLAGS.data_type == 'position':
        assert (FLAGS.num_channels == 2)
    else:
        raise ValueError('invalid data type')

    if FLAGS.data_dir == '' or not os.path.exists(FLAGS.data_dir):
        raise ValueError('invalid data directory {}'.format(FLAGS.data_dir))

    train_data_path = os.path.join(FLAGS.data_dir, 'quantum-train.tfr')

    if FLAGS.output_dir == '':
        raise ValueError('invalid output directory {}'.format(FLAGS.output_dir))
    elif not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    event_log_dir = os.path.join(FLAGS.output_dir, '')
    checkpoint_path = os.path.join(FLAGS.output_dir, 'model.ckpt')

    summ = Summaries()

    model = Model(act = FLAGS.activation, pool = FLAGS.pooling, with_memory = FLAGS.with_memory)
    assert FLAGS.train_batch_size > 0, 'invalid training batch size.'

    cur_iter = tf.placeholder(tf.float32, [], name = "cur_iter")
    train_loss, train_op, train_summ_op = train(model, train_data_path, cur_iter, summ = summ)

    print("{} model validation".format('execute' if FLAGS.validate else 'skip'))
    if FLAGS.validate:
        assert FLAGS.xval_batch_size > 0, 'batch size for cross validation'
        xval_data_path = os.path.join(FLAGS.data_dir, 'quantum-val.tfr')
        xval_op, xval_summ_op = xval(model, xval_data_path, summ = summ)

    saver = tf.train.Saver(tf.trainable_variables() + tf.moving_average_variables())

    writer = tf.summary.FileWriter(event_log_dir, tf.get_default_graph())

    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)

    assert (FLAGS.gpus != ''), 'invalid GPU specification'
    config.gpu_options.visible_device_list = FLAGS.gpus

    # Train.

    with tf.Session(config = config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # start_iteration = sess.run(global_step)
        start_iteration = 0

        for train_iteration in range(start_iteration, FLAGS.num_iterations):
            _, loss, summ_op_str = sess.run([train_op, train_loss, train_summ_op],
                                            feed_dict = {cur_iter: train_iteration})
            print("{}: training loss {}".format(train_iteration, loss))
            writer.add_summary(summ_op_str, train_iteration)

            if FLAGS.validate and train_iteration % FLAGS.report_interval == 1:
                summ_xval_str = sess.run(xval_summ_op)
                writer.add_summary(summ_xval_str, train_iteration) 

        tf.logging.info('Saving model.')
        saver.save(sess, checkpoint_path)
        tf.logging.info('Training complete')

    writer.close()


if __name__ == "__main__":
    tf.app.run()
