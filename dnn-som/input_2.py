'''
@author: MSUser
'''
import os
import numpy as np
import tensorflow as tf
import sonnet as snt 

class Input(snt.AbstractModule):
    def __init__(self, batch_size, image_shape, num_epochs = -1, name = 'input'):
        '''
        Args:
            batch_size: number of tfrecords to dequeue
            img_shape: the expected shape of series of images
        '''
        super(Input, self).__init__(name = name)
        self._batch_size = batch_size
        self._img_shape = image_shape
        self._num_epochs = num_epochs

    def _parse_function(self, example):
        dims = np.prod(self._img_shape)

        features = {
            "image": tf.FixedLenFeature([dims], dtype = tf.float32),
            "label": tf.FixedLenFeature([], dtype = tf.int64)
        }

        example_parsed = tf.parse_single_example(serialized = example,
                                                 features = features)

        image = tf.reshape(example_parsed['image'], self._img_shape)
        label = example_parsed['label']

        return image, label
        
    def _build(self, filenames):
        '''
        Retrieve tfrecord from files and prepare for batching dequeue
        Args:
            tfr_ptn: 
        Returns:
            image series and label in batch
            
        '''

        dataset = tf.data.TFRecordDataset(filenames)

        dataset = dataset.map(self._parse_function)

        dataset = dataset.repeat(1)
        dataset = dataset.batch(self._batch_size)

        iterator = dataset.make_initializable_iterator()
        with tf.control_dependencies([iterator.initializer]):
            images, labels = iterator.get_next()
       
        return images, labels
    
if __name__ == '__main__':
    from config import FLAGS   
 
    filenames = tf.placeholder(tf.string, shape=[None])
    input_ = Input(1, [FLAGS.img_height, FLAGS.img_width, FLAGS.num_channels])
    images, labels = input_(filenames)

    with tf.Session() as sess:

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        
        images_val, labels_val = sess.run([images, labels], feed_dict = {filenames: ["C_0_0.tfr"]})

        print(images_val.shape)
        print(labels_val)

        '''
        from scipy import io
        io.savemat("f1_real.mat", {"f1_real": images_val[0, :, :, 0]})
        '''
