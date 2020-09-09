import os
import sys
import psutil
import time
import numpy as np


def log(msg, verbose=True):
    if verbose:
        asctime = time.asctime(time.localtime())
        print(f'{asctime} > {msg}')


def one_hot(array, n_classes):
    return np.eye(n_classes)[array]


def restart():
    try:
        p = psutil.Process(os.getpid())
        for handler in p.open_files() + p.connections():
            os.close(handler.fd)
    except Exception as e:
        raise e
    python = sys.executable
    os.execl(python, python, *sys.argv)
