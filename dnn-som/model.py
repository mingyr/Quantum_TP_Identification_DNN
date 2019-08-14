import tensorflow as tf
import sonnet as snt
from som import SOM
from utils import Pooling, Activation

from config import FLAGS

class Model(snt.AbstractModule):
    def __init__(self, act = None, pool = None, with_memory = True, 
                 summ = None, residual = True, log = False, name = "model"):
        super(Model, self).__init__(name = name)

        self._with_memory = with_memory
        self._summ = summ
        self._residual = residual
        self._num_blocks = 6

        self._log = log

        with self._enter_variable_scope():
            self._act = Activation(act, verbose = True)
            self._pool = Pooling(pool, padding = 'VALID', verbose = True) 

            if self._residual:
                self._convs = [snt.Conv2D(eval("FLAGS.num_outputs_block_%d" % (i + 1)), FLAGS.filter_size, padding = snt.VALID, use_bias = False)
                                  for i in range(self._num_blocks)]

                self._sepconvs = [snt.SeparableConv2D(eval("FLAGS.num_outputs_block_%d" % (i + 1)), 1, FLAGS.filter_size, padding = snt.SAME, use_bias = False)
                                  for i in range(self._num_blocks)]
            else:
                self._sepconvs = [snt.SeparableConv2D(eval("FLAGS.num_outputs_block_%d" % (i + 1)), 1, FLAGS.filter_size, padding = snt.VALID, use_bias = False)
                                  for i in range(self._num_blocks)]

            self._seq = snt.Sequential([snt.Linear(output_size = FLAGS.num_outputs_dense), tf.nn.relu,
                                        snt.Linear(output_size = FLAGS.num_classes)])

            if self._with_memory:
                print("Model with memory enabled")

                config = \
                {
                    "height": FLAGS.memory_height,
                    "width": FLAGS.memory_width,
                    "input_size": 32, # very dangeous, hard-coded
                    "num_iters": FLAGS.num_iterations,
                    "learning_rate": FLAGS.lr_som
                }

                self._som = SOM(**config)

    def _build(self, inputs, training = True, cur_iter = None):
        outputs = tf.identity(inputs)

        for i in range(self._num_blocks):
            outputs = self._pool(outputs)
            if self._residual:
                outputs = self._convs[i](outputs)
                outputs = self._sepconvs[i](outputs) + outputs
            else:
                outputs = self._sepconvs[i](outputs)

            outputs = self._act(outputs)

            '''
            if training:
                self._summ.register('train', self._batch_norms[i].moving_mean.name, self._batch_norms[i].moving_mean)
                self._summ.register('train', self._batch_norms[i].moving_variance.name, self._batch_norms[i].moving_variance)
            '''

        cnn_outputs = snt.BatchFlatten()(outputs)

        if self._with_memory:
            som_outputs = tf.stop_gradient(self._som(cnn_outputs, training, cur_iter, self._log))
            return self._seq(tf.concat([cnn_outputs, som_outputs], axis = -1))
        else:
            return self._seq(cnn_outputs)

def test():
    from config import FLAGS

    t = tf.truncated_normal([16, 599, 599, 4])

    model = Model(act = FLAGS.activation, pool = FLAGS.pooling, with_memory = FLAGS.with_memory)
    r = model(t, True, 0)

    writer = tf.summary.FileWriter('model', tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        v = sess.run(r)

        print(v.shape)
   
    writer.close()

if __name__ == '__main__':
    test()

