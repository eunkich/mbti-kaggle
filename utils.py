import os
import sys
import psutil
import time
import numpy as np
import pandas as pd


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


def save_experiment_results(dataframe, args,
                            filename='./result_table.csv'):
    # add arguments
    args_dict = vars(args)
    columns = list(dataframe.columns) + list(args_dict.keys())
    result = pd.DataFrame(
        index=dataframe.index,
        columns=columns
    )
    result[list(args_dict.keys())] = list(args_dict.values())
    result[dataframe.columns] = dataframe.values

    # check if file already exists
    pardir = os.path.abspath(os.path.join(filename, os.pardir))
    os.makedirs(pardir, exist_ok=True)
    if os.path.isfile(filename):
        result_ = pd.read_csv(filename, index_col=0)
        result_ = result_.append(result)
        result_.to_csv(filename)
    else:
        result.to_csv(filename)
