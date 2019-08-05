import os
import glob
import time
import numpy as np
import pickle
import warnings
import matlab.engine

# data path to where the raw data generated
datapath = 'F:\\RawData_16July\\RawData_momentum'
# datapath = 'F:\\RawData_16July\\RawData_position'  
# datapath = 'F:\Transition_18_may_2019\RawData_momentum_transition'
# datapath = 'F:\Transition_18_may_2019\\RawData_position_transition'

print("datapath -> {}".format(datapath))

filenames = {'train_filenames': [],
             'val_filenames': [],
             'test_filenames': []}

listname = os.listdir(datapath)

eng = matlab.engine.start_matlab()

for C in listname:
    sub_datapath = os.path.join(datapath, C, "*.mat")
    print(sub_datapath)
    sub_listname = glob.glob(sub_datapath)

    num_filenames = len(sub_listname)
    print("Chern number = {}, number of files {}".format(C, num_filenames))
    
    indices = list(range(num_filenames))
    eng.rng('shuffle','combRecursive')
    
    p = np.array(eng.randperm(num_filenames))
    p = p.astype(int) - 1
    
    indices = [indices[i] for i in p.reshape([-1])] 

    pivot = int(round(num_filenames * 0.1))

    val_indices = range(pivot)
    test_indices = range(pivot, 2 * pivot)

    for j in range(num_filenames):
        filename = sub_listname[indices[j]]

        if j in val_indices:
            filenames['val_filenames'].append(filename)
        elif j in test_indices:
            filenames['test_filenames'].append(filename)
        else:
            filenames['train_filenames'].append(filename)

print('#training samples {}'.format(len(filenames['train_filenames'])))
print('#validation samples {}'.format(len(filenames['val_filenames'])))
print('#test samples {}'.format(len(filenames['test_filenames'])))

with open('filenames_momentum.dat', 'wb') as f:
    warnings.warn("Check datapath = 'F:\\RawData_16July\\RawData_momentum'", UserWarning)
    pickle.dump(filenames, f)

'''   
with open('filenames_position.dat', 'wb') as f:
    warnings.warn("Check datapath = 'F:\\RawData_16July\\RawData_position'", UserWarning)
    pickle.dump(filenames, f)
''' 

'''
with open('filenames_transition_momentum3.dat', 'wb') as f:
    warnings.warn("Check datapath = 'F:\\RawData_16July\\RawData_momentum_transition'", UserWarning)
    pickle.dump(filenames, f)
'''

'''
with open('filenames_transition_position3-tmp.dat', 'wb') as f:
    warnings.warn("Check datapath = 'F:\\RawData_16July\\RawData_position_transition'", UserWarning)
    pickle.dump(filenames, f)

eng.quit()
'''
