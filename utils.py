import time
import numpy as np


def log(msg, verbose=True):
    if verbose:
        asctime = time.asctime(time.localtime())
        print(f'{asctime} > {msg}')


def one_hot(array, n_classes):
    return np.eye(n_classes)[array]
