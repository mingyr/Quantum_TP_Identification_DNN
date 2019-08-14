import os
import re
import subprocess
import signal
import string
from monitor import Parser, Cmd
import shlex

def train(parse = False, pseudo = False):

    num_iters = 1

    for i in range(num_iters):

        cmdline_str = 'python train_classification.py --type=classification --train_batch_size=32 --xval_batch_size=32 --data_dir=/data/yuming/quantum-processed-data --output_dir=output/{} --gpus=0,1'.format(i)
        print(cmdline_str)

        if pseudo:
            continue

        cmdline = shlex.split(cmdline_str)

        if parse:
            parser = Parser().get_custom_parser("\\#trials\\:\\d+")
            parser = Parser().get_custom_parser(".+")
            cmd = Cmd(parser)
            cmd.start(cmdline, "tmp")
            cmd.wait()
        else:
            cmd = Cmd()
            cmd.start(cmdline)
            cmd.wait()


if __name__ == '__main__':
    train(pseudo = True)

