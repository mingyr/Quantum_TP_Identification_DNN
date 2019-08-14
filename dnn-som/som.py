import numpy as np
import warnings
import tensorflow as tf
import sonnet as snt
import math

from tensorflow.python import debug as tf_debug

class SOM(snt.AbstractModule):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
    def __init__(self, height, width, input_size, num_iters, learning_rate, name = "som"):
        """
        height: height of the lattice
        width: width of the lattice
        num_iters: an integer denoting the number of iterations undergone while training.
        input_size: the dimensionality of the training inputs.
        learning_rate: a number denoting the initial time(iteration no)-based learning rate. Default value is 0.1
        """
        super(SOM, self).__init__(name = name)

        #To check if the SOM has been trained
        self._trained = False

        #Assign required variables first
        self._height = height
        self._width = width
        self._radius = max(height / 2.0, width / 2.0)
        self._input_size = input_size
        self._learning_rate = learning_rate
        self._num_iters = num_iters
        self._time_constant = num_iters / math.log(self._radius)

        print("SOM height: {}".format(self._height))
        print("SOM width: {}".format(self._width))
        print("SOM initial radius {}".format(self._radius))
        print("SOM input size {}".format(self._input_size))
        print("SOM learning rate {}".format(self._learning_rate))
        print("SOM number of iterations {}".format(self._num_iters))
        print("SOM time constant {}".format(self._time_constant))

        with self._enter_variable_scope():

            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE

            #Randomly initialized weightage vectors for all neurons,
            #stored together as a matrix Variable of size [height * width, input_size]
            self._weights = tf.get_variable("w", [height * width, input_size], tf.float32,
                                            initializer = tf.random_normal_initializer())
                                            # initializer = tf.zeros_initializer())
            #shape(self._weights) = (M, L)

            #Matrix of size [height * width, 2] for SOM grid locations of neurons
            self._locations = tf.constant([[i, j] for i in range(height) for j in range(width)])
            #shape(self._locations) = (M, 2)

    @property
    def trained(self):
        return self._trained

    @trained.setter
    def trained(self, value):
        self._trained = value

    def _train(self, inputs):

        ##CONSTRUCT TRAINING OP PIECE BY PIECE
        #Only the final, 'root' training op needs to be assigned as
        #an attribute to self, since all the rest will be executed
        #automatically during training

        #To compute the Best Matching Unit given a vector
        #Basically calculates the Euclidean distance between every
        #neuron's weightage vector and the input, and returns the
        #index of the neuron which gives the least value
        inputs = tf.expand_dims(inputs, axis = 1)
        #shape(inputs) = (N, 1, L)

        weights = tf.expand_dims(self._weights, axis = 0)
        #shape(weights) = (1, M, L)

        bmu_index = tf.argmin(tf.reduce_sum(tf.square(inputs - self._weights), -1), axis = 1, name = "bmu_index")
        #shape(bmu_index) = (N)

        #This will extract the location of the BMU based on the BMU's index
        bmu_loc = tf.gather(self._locations, bmu_index, name = "bmu_loc")
        #shape(bmu_loc) = (N, 2)

        # summaries.append(tf.summary.text("bmu", tf.as_string(bmu_loc)))
        # summaries.append(tf.summary.scalar("learning_rate", learning_rate))
        # summaries.append(tf.summary.scalar("neigh_radius", neigh_radius))

        #Construct the op that will generate a vector with learning rates
        #for all neurons, based on iteration number and location wrt BMU

        bmu_loc = tf.expand_dims(bmu_loc, axis = 1)
        #shape(bmu_loc) = (N, 1, 2)

        bmu_distance_square = tf.cast(tf.reduce_sum(tf.square(self._locations - bmu_loc), -1, True), tf.float32, name = "bmu_distance_square")
        #shape(bmu_distance_square) = (N, M, 1)

        neigh_radius_square = tf.square(self._neigh_radius, name = "neigh_radius_square")

        bmu_distance_mask = tf.nn.relu(tf.sign(neigh_radius_square - bmu_distance_square), name = "bmu_distance_mask")
        #shape(bmu_distance_mask) = (N, M, 1)

        # summaries.append(tf.summary.histogram("bmu_distance_mask", bmu_distance_mask))

        learning_efficiency = tf.multiply(self._effective_learning_rate,
            tf.exp(-0.5 * tf.cast(bmu_distance_square, tf.float32) / neigh_radius_square), name = "learning_efficiency")
        #shape(learning_efficiency) = (N, M, 1)

        effective_learning_efficiency = tf.multiply(learning_efficiency, tf.cast(bmu_distance_mask, tf.float32), name = "effective_learning_efficiency")
        #shape(effective_learning_efficiency) = (N, M, 1)

        #Finally, the op that will use learning_rate_op to update
        #the weightage vectors of all neurons based on a particular input

        # summaries_op = tf.summary.merge(summaries)

        delta = tf.subtract(inputs, self._weights, name = "weight_delta")
        #shape(delta) = (N, M, L)

        dummy_op = tf.reduce_mean(effective_learning_efficiency * delta, axis = 0)

        new_weights = tf.add(self._weights, \
            tf.reduce_mean(effective_learning_efficiency * delta, axis = 0), name = "new_weights")
        #shape(new_weights) = (M, L)

        train_op = tf.assign(self._weights, new_weights, name = "train_op")

        with tf.control_dependencies([train_op]):
            outputs = tf.gather(self._weights, bmu_index)

        return outputs

    def _test(self, inputs, log = False):
        inputs = tf.expand_dims(inputs, axis = 1)
        #shape(inputs) = (N, 1, L)

        weights = tf.expand_dims(self._weights, axis = 0)
        #shape(weights) = (1, M, L)

        bmu_index = tf.argmin(tf.reduce_sum(tf.square(inputs - self._weights), -1), axis = 1, name = "bmu_index")
        #shape(bmu_index) = (N)

        if log:
            debug_op = tf.Print(inputs, [bmu_index], first_n = 1000, summarize = 1000)
            with tf.control_dependencies([debug_op]):
                outputs = tf.gather(self._weights, bmu_index)
        else:
            outputs = tf.gather(self._weights, bmu_index)

        return outputs

    def _build(self, inputs, is_training = True, cur_iter = -1.0, log = False):
        #shape(inputs) = (N, L)

        summaries = []

        if is_training:
            #To compute the current learning rate and neighbourhood values based on current iteration number
            with tf.control_dependencies([tf.assert_greater_equal(cur_iter, 0.0)]):
                self._effective_learning_rate = tf.multiply(self._learning_rate, tf.exp(-cur_iter / self._num_iters), name = "learning_rate")
                self._neigh_radius = tf.multiply(self._radius, tf.exp(-cur_iter / self._time_constant), name = "neigh_radius")

            outputs = self._train(inputs)            
        else:
            outputs = self._test(inputs, log)

        return outputs

    @property
    def weights(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            warnings.warn("SOM not trained yet")
        return self._weights

    def bmu(self, inputs):
        """
        Maps input vector to the relevant neuron in the SOM grid.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")

        index = tf.argmin(tf.reduce_sum(tf.square(inputs - self._weights), -1))
        index = [index // self._height, index % self._width]
        return index

    def image(self):
        if self._input_size != 1 and self._input_size != 3:
            raise AttributeError("input_size should be 1 or 3 for later retriving weighting image")
        min_elem = tf.reduce_min(tf.reduce_min(self._weights))
        max_elem = tf.reduce_max(tf.reduce_max(self._weights))
        output = (self._weights - tf.reshape(min_elem, [1, 1])) / tf.reshape((max_elem - min_elem), [1, 1])

        return tf.reshape(output, [-1, self._height, self._width, self._input_size])
                

def test(debug = False):
    from random import randint

    """
    Trains the SOM.
    'input_vects' should be an iterable of 1-D NumPy arrays with
    dimensionality as provided during initialization of this SOM.
    Current weightage vectors for all neurons(initially random) are
    taken as starting conditions for training.
    """

    colors = [
        [
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 0., 0.5],
        ],
        [
            [0.125, 0.529, 1.0],
            [0.33, 0.4, 0.67],
            [0.6, 0.5, 1.0],
        ],
        [
            [0., 1., 0.],
            [1., 0., 0.],
            [0., 1., 1.],
        ],
        [
            [1., 0., 1.],
            [1., 1., 0.],
            [1., 1., 1.],
        ],
        [
            [.33, .33, .33],
            [.5, .5, .5],
            [.66, .66, .66]
        ]
    ]
    
    inputs = tf.placeholder(tf.float32, [None, 3, 3], name = "inputs")
    cur_iter = tf.placeholder(tf.float32, [], name = "cur_iter")

    config = {
        "height": 256,
        "width": 256,
        "input_size": 3,
        "num_iters": 5000,
        "learning_rate": 0.1
    } 

    som = SOM(**config)

    is_training = True
    outputs = som(inputs, is_training, cur_iter)

    if is_training:
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep = 0)

        add_img_op_1 = tf.summary.image("initial weights image", som.image())
        add_img_op_2 = tf.summary.image("final weights image", som.image())
 
        writer = tf.summary.FileWriter("output", tf.get_default_graph())

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            if debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            for i in range(config["num_iters"]):
                print("iter: {}".format(i))
                outputs_val = sess.run(outputs, feed_dict = {inputs: colors, cur_iter: i})

                if i == 0:
                    writer.add_summary(sess.run(add_img_op_1), i)

            print("outputs_val -> {}".format(outputs_val))

            writer.add_summary(sess.run(add_img_op_2), i)

            som.trained = True

            saver.save(sess, "output/model.ckpt")

            for c in colors:
                for e in c:
                    print("color {} <-> center {}".format(e, sess.run(som.bmu(e))))

        writer.close()
    else:
        saver = tf.train.Saver(tf.trainable_variables())
   
        ckpt = tf.train.get_checkpoint_state('output')
        with tf.Session() as sess:
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint.
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

            som.trained = True 

            for c in colors:
                for e in c:
                    print("color {} <-> center {}".format(e, sess.run(som.bmu(e))))

if __name__ == "__main__":

    from optparse import OptionParser

    parser = OptionParser(usage="usage: python som.py [-d|--debug]",
                          version="som.py 1.0")
    parser.add_option("-d", "--debug",
                      action = "store_true",
                      dest = "debug",
                      default = False,
                      help = "debug the script using tfdbg")

    (options, args) = parser.parse_args()

    test(options.debug)


