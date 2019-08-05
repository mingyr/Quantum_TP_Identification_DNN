import pickle
import scipy.io 

with open('filenames_momentum.dat', 'rb') as f:
    filenames = pickle.load(f)

train_filenames = filenames['train_filenames']
val_filenames = filenames['val_filenames']
test_filenames = filenames['test_filenames']

scipy.io.savemat('filenames_momentum.mat', {'train_filenames': train_filenames,
                                             'val_filenames': val_filenames,
                                             'test_filenames': test_filenames})
