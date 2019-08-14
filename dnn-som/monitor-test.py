import os
import re
import subprocess
import signal
import string
import shlex
from monitor import Parser, Cmd

def test_eeg_learn():
    results = []

    num_iters = 5

    folds = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15]
    test_samples = [185, 212, 199, 201, 196, 201, 193, 202, 210, 225, 217, 209, 220]

    for i, d in enumerate(folds):
        for j in range(num_iters):

            cmdline_str = 'python test.py --type=classification --test_batch_size={} --data_dir=/data/yuming/eeg-processed-data/eeg-learn/topography-tfr/{} --output_dir=output-dnc/{}/{} --gpus=0'.format(test_samples[i], d, d, j)
            print(cmdline_str)

            cmdline = shlex.split(cmdline_str)

            # parser = Parser().get_custom_parser("\\#trials\\:\\d+")
            # parser = Parser().get_custom_parser(".+")
'''
            parser = Parser().get_custom_parser("(accu \\-> (\\d|\\.)+)|(loss \\-> (\\d|\\.)+)")
            cmd = Cmd(parser, results)
            cmd.start(cmdline, "tmp")
            cmd.wait()

    print("")
    for r in results:
        print(r)

'''


def test_lane_keeping(parse = True, pseudo = False):
    results = []

    num_iters = 5

    '''
    "convpool_conv1d"
    "convpool_lstm"
    "convpool_mix"
    '''

    test_samples = [646, 628, 908, 734, 891, 760, 1267]

    for i in range(len(test_samples) - 1):
        for j in range(num_iters):
            cmdline_str = 'python test.py --model=convpool_mix --type=regression --test_batch_size={} --data_dir=/data/yuming/eeg-processed-data/lane-keeping/full-channel-topo/{} --output_dir=output-lane-keeping/mix/output-5/{}/{} --gpus=2'.format(test_samples[i], i, i, j)
            print(cmdline_str)

            if pseudo:
                continue

            cmdline = shlex.split(cmdline_str)

            # parser = Parser().get_custom_parser("\\#trials\\:\\d+")
            # parser = Parser().get_custom_parser(".+")
            parser = Parser().get_custom_parser("(accu \\-> (\\d|\\.)+)|(loss \\-> (\\d|\\.)+)")
            cmd = Cmd(parser, results)
            cmd.start(cmdline, "tmp")
            cmd.wait()
 
    print("\n")
    for res in results:
        print(res)

def test_lane_keeping_sess(parse = True, pseudo = False):
    results = []

    num_iters = 5

    '''
    "convpool_conv1d"
    "convpool_lstm"
    "convpool_mix"
    '''

    # test_samples = [646, 628, 908, 734, 891, 760, 1267]

    test_samples = [327, 139, 276, 515, 426]

    test_dirs = ['s02_050921m', 's04_051024m', 's05_051120m', 's13_060213m', 's22_091006m']

    for i in range(len(test_samples)):
        for j in range(num_iters):
            cmdline_str = 'python test.py --model=convpool_lstm --type=regression --test_batch_size={} --data_dir=/data/yuming/eeg-processed-data/lane-keeping/full-channel-topo-sess/{} --output_dir=output-lane-keeping/lstm/output-5/{}/{} --gpus=0'.format(test_samples[i], test_dirs[i], i, j)
            print(cmdline_str)

            if pseudo:
                continue

            cmdline = shlex.split(cmdline_str)

            if parse:
                # parser = Parser().get_custom_parser("\\#trials\\:\\d+")
                # parser = Parser().get_custom_parser(".+")
                parser = Parser().get_custom_parser("(accu \\-> (\\d|\\.)+)|(loss \\-> (\\d|\\.)+)")
                cmd = Cmd(parser, results)
                cmd.start(cmdline, "tmp")
                cmd.wait()
            else:
                cmd = Cmd()
                cmd.start(cmdline)
                cmd.wait()
                
    if parse:
        print("\n")
        for res in results:
            print(res)

def test_lane_keeping_2(parse = True, pseudo = False):
    results = []

    num_iters = 5

    '''
    "convpool_conv1d"
    "convpool_lstm"
    "convpool_mix"
    '''

    test_samples = [480, 443, 372, 496, 476]

    for i in range(len(test_samples)):
        for j in range(num_iters):
            cmdline_str = 'python test.py --model=convpool_mix --type=regression --test_batch_size={} --data_dir=/data/yuming/eeg-processed-data/lane-keeping/full-channel-topo-motionless/{} --output_dir=output-lane-keeping-motionless/mix/output-1/{}/{} --gpus=1'.format(test_samples[i], i, i, j)
            print(cmdline_str)

            if pseudo:
                continue

            cmdline = shlex.split(cmdline_str)

            if parse:
                # parser = Parser().get_custom_parser("\\#trials\\:\\d+")
                # parser = Parser().get_custom_parser(".+")
                parser = Parser().get_custom_parser("(accu \\-> (\\d|\\.)+)|(loss \\-> (\\d|\\.)+)")
                cmd = Cmd(parser, results)
                cmd.start(cmdline, "tmp")
                cmd.wait()
            else:
                cmd = Cmd()
                cmd.start(cmdline)
                cmd.wait()

    if parse:
        for res in results:
            print(res)


if __name__ == '__main__':
    test_lane_keeping_2(parse = False, pseudo = False)
    

