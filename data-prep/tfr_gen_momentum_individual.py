import os
import tensorflow as tf
import matlab.engine
import random 
import numpy as np

label_map = {'C_0': 2,  'C_1': 3,  'C_-1': 1,  'C_2': 4,  'C_-2': 0}
label_inverse_map = {2: 'C_0',  3: 'C_1',  1: 'C_-1', 4: 'C_2',  0: 'C_-2'}

eng = matlab.engine.start_matlab("-nojvm")

# filename = '/data/yuming/quantum-data/RawData_momentum/C_-1/m_22.00_t1y_1_t2_1.0_t3_10.00.mat'
# filename = '/data/yuming/quantum-data/RawData_momentum/C_-2/m_-0.02_t1y_1_t2_1.0_t3_-0.02.mat'
# filename = '/data/yuming/quantum-data/RawData_momentum/C_0/m_-22.00_t1y_1_t2_1.0_t3_0.05.mat'
# filename = '/data/yuming/quantum-data/RawData_momentum/C_1/m_-6.26_t1y_1_t2_1.0_t3_7.44.mat'
# filename = '/data/yuming/quantum-data/RawData_momentum/C_2/m_0.02_t1y_-1_t2_1.0_t3_-0.21.mat'


# filename = '/data/yuming/quantum-data-individual/momentum/C_0/m_-19.29_t1y_1_t2_5.0_t3_0.00.mat'
# filename = '/data/yuming/quantum-data-individual/momentum/C_0/m_-19.29_t1y_1_t2_5.0_t3_-0.20.mat'
# filename = '/data/yuming/quantum-data-individual/momentum/C_0/m_-19.29_t1y_1_t2_5.0_t3_0.20.mat'

# filename = '/data/yuming/quantum-data-individual/momentum/C_1/m_-19.39_t1y_1_t2_5.0_t3_-18.87.mat'
# filename = '/data/yuming/quantum-data-individual/momentum/C_1/m_-19.39_t1y_1_t2_5.0_t3_-19.43.mat'
# filename = '/data/yuming/quantum-data-individual/momentum/C_1/m_-19.39_t1y_1_t2_5.0_t3_19.43.mat'

# filename = '/data/yuming/quantum-data-individual/momentum/C_-1/m_10.77_t1y_1_t2_5.0_t3_-18.73.mat'
# filename = '/data/yuming/quantum-data-individual/momentum/C_-1/m_10.77_t1y_1_t2_5.0_t3_-19.36.mat'
# filename = '/data/yuming/quantum-data-individual/momentum/C_-1/m_10.77_t1y_1_t2_5.0_t3_19.36.mat'

# filename = '/data/yuming/quantum-data-individual/momentum/C_2/m_-5.12_t1y_-1_t2_5.0_t3_-0.10.mat'
# filename = '/data/yuming/quantum-data-individual/momentum/C_2/m_-5.12_t1y_-1_t2_5.0_t3_0.10.mat'
# filename = '/data/yuming/quantum-data-individual/momentum/C_2/m_-5.12_t1y_-1_t2_5.0_t3_0.30.mat'

# filename = '/data/yuming/quantum-data-individual/momentum/C_-2/m_-5.12_t1y_1_t2_5.0_t3_-0.10.mat'
# filename = '/data/yuming/quantum-data-individual/momentum/C_-2/m_-5.12_t1y_1_t2_5.0_t3_0.10.mat'
filename = '/data/yuming/quantum-data-individual/momentum/C_-2/m_5.12_t1y_1_t2_5.0_t3_-0.20.mat'


label = label_map[filename.split('/')[-2]]

def float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

f1_placeholder = tf.placeholder(tf.complex64, [599, 599])
f2_placeholder = tf.placeholder(tf.complex64, [599, 599])

f1_real = tf.expand_dims(tf.real(f1_placeholder), axis = -1)
f1_imag = tf.expand_dims(tf.imag(f1_placeholder), axis = -1)
f2_real = tf.expand_dims(tf.real(f2_placeholder), axis = -1)
f2_imag = tf.expand_dims(tf.imag(f2_placeholder), axis = -1)

f_example = tf.concat([f1_real, f1_imag, f2_real, f2_imag], axis = -1)

def gen_tfr(filename, sess, writer):
    data = eng.load(filename)
    f1 = np.array(data['f1'], np.complex64)
    f2 = np.array(data['f2'], np.complex64)

    f_example_val = sess.run(f_example, feed_dict = {f1_placeholder: f1, f2_placeholder: f2})

    feature = {
        'image': float_feature(np.reshape(f_example_val, [-1])),
        'label': int64_feature(label)
    }

    example = tf.train.Example(features = tf.train.Features(feature = feature))

    writer.write(example.SerializeToString())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    save_path = ''

    print('beginning prepare sample tfrecords')
    writer = tf.python_io.TFRecordWriter(os.path.join(save_path, '{}.tfr'.format(label_inverse_map[label])))
    gen_tfr(filename, sess, writer)
    writer.close()
    print('end of sample tfrecords preparation')
        
eng.quit()


