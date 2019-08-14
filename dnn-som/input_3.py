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
            "coord": tf.FixedLenFeature([2], dtype = tf.float32)
        }

        example_parsed = tf.parse_single_example(serialized = example,
                                                 features = features)

        image = tf.reshape(example_parsed['image'], self._img_shape)
        coord = example_parsed['coord']

        return image, coord
        
    def _build(self, filenames):
        '''
        Retrieve tfrecord from files and prepare for batching dequeue
        Args:
            tfr_ptn: 
        Returns:
            image series and label in batch
            
        '''

        assert os.path.isfile(filenames), "invalid file path: {}".format(filenames)

        if type(filenames) == list:
            dataset = tf.data.TFRecordDataset(filenames)
        elif type(filenames) == str:
            dataset = tf.data.TFRecordDataset([filenames])
        else:
            raise ValueError('wrong type {}'.format(type(filenames)))

        dataset = dataset.map(self._parse_function)

        dataset = dataset.batch(self._batch_size)
        dataset = dataset.repeat(self._num_epochs)

        iterator = dataset.make_one_shot_iterator()
        images, coords = iterator.get_next()
       
        return images, coords
    
if __name__ == '__main__':
    from config import FLAGS   
 
    input_ = Input(64, [FLAGS.img_height, FLAGS.img_width, FLAGS.num_channels])
    images, coords = input_('/scratch/quantum-data/interact/phi_0.01_momentum/quantum.tfr')

    with tf.Session() as sess:

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        
        images_val, coords_val = sess.run([images, coords])

        print(images_val.shape)
        print(coords_val)

        '''
        from scipy import io
        io.savemat("f1_real.mat", {"f1_real": images_val[0, :, :, 0]})
        '''
