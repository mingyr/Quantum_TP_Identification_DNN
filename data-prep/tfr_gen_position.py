import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import matlab.engine
import random 
import numpy as np
import pickle

label_map = {'C_0': 2,  'C_1': 3,  'C_-1': 1,  'C_2': 4,  'C_-2': 0}
label_inverse_map = {2: 'C_0',  3: 'C_1',  1: 'C_-1', 4: 'C_2',  0: 'C_-2'}

eng = matlab.engine.start_matlab("-nojvm")

with open('/scratch/quantum-meta/filenames_position.dat', 'rb') as f:
    filenames = pickle.load(f)

train_filenames = filenames['train_filenames']
val_filenames = filenames['val_filenames']
test_filenames = filenames['test_filenames']

num_training = len(train_filenames)
num_validation = len(val_filenames)
num_test = len(test_filenames)

print('#training samples -> {}'.format(num_training))
print('#validation samples -> {}'.format(num_validation))
print('#test samples -> {}'.format(num_test))

train_filename_indices = list(range(num_training))
val_filename_indices = list(range(num_validation))
test_filename_indices = list(range(num_test))

random.shuffle(train_filename_indices)
random.shuffle(val_filename_indices)
random.shuffle(test_filename_indices)

def float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

kup_placeholder = tf.placeholder(tf.float64, [599, 599])
kdown_placeholder = tf.placeholder(tf.float64, [599, 599])

k_example = tf.concat([tf.expand_dims(kup_placeholder, axis = -1),
                       tf.expand_dims(kdown_placeholder, axis = -1)], axis = -1)

def gen_tfr(filename, sess, writer):
    prefix='/scratch'
    ss = filename.split('\\')
    filename = os.path.join(prefix, '/'.join(ss[1:]))
    assert(os.path.isfile(filename)), 'invalid file path'

    data = eng.load(filename)
    kup = np.array(data['kup'], np.float64)
    kdown = np.array(data['kdown'], np.float64)

    k_example_val = sess.run(k_example, feed_dict = {kup_placeholder: kup, kdown_placeholder: kdown})
    label = label_map[filename.split('/')[-2]]

    feature = {
        'image': float_feature(np.reshape(k_example_val, [-1])),
        'label': int64_feature(label)
    }

    example = tf.train.Example(features = tf.train.Features(feature = feature))

    writer.write(example.SerializeToString())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    save_path = '/scratch/quantum-data-normal/position/1'

    print('beginning prepare training tfrecords')
    writer = tf.python_io.TFRecordWriter(os.path.join(save_path, 'quantum-train.tfr'))

    for i in range(num_training):
        print('processing {} / {}'.format(i, num_training))
        filename = train_filenames[train_filename_indices[i]]
        gen_tfr(filename, sess, writer)
    writer.close()
    print('end of training tfrecords preparation')

    print('beginning prepare validation tfrecords')
    writer = tf.python_io.TFRecordWriter(os.path.join(save_path, 'quantum-val.tfr'))

    for i in range(num_validation):
        print('processing {} / {}'.format(i, num_validation))
        filename = val_filenames[val_filename_indices[i]]
        gen_tfr(filename, sess, writer)
    writer.close()
    print('end of validation tfrecords preparation')

    print('beginning prepare test tfrecords')
    writer = tf.python_io.TFRecordWriter(os.path.join(save_path, 'quantum-test.tfr'))

    for i in range(num_test):
        print('processing {} / {}'.format(i, num_test))
        filename = test_filenames[test_filename_indices[i]]
        gen_tfr(filename, sess, writer)
    writer.close()
    print('end of test tfrecords preparation')
        
eng.quit()


