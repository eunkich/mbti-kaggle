import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
from train import ensemble


def kaggle(args):
    """Reproduce the original preprocessing from Kaggle"""
    from data.corpus import load_kaggle
    from data.preprocess import CountVectorizer

    posts, types = load_kaggle(verbose=args.verbose)
    loader = CountVectorizer(posts, types, args)
    ensemble(loader, args)


def kaggle_masked(args):
    from data.corpus import load_kaggle_masked
    from data.preprocess import CountVectorizer

    posts, types = load_kaggle_masked(verbose=args.verbose)
    loader = CountVectorizer(posts, types, args)
    ensemble(loader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MBTI classification from kaggle dataset"
    )
    tasks = ['kaggle', 'kaggle_masked']
    parser.add_argument("--task", type=str, choices=tasks)
    parser.add_argument("--seed", type=str, default=-1)
    parser.add_argument('-q', "--quiet", action="store_true")
    parser.add_argument('-o', "--output", type=str, default='result.csv')

    parser.add_argument_group("Preprocessing options")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--max_features", type=int, default=1500)
    parser.add_argument("--max_df", type=float, default=0.5)
    parser.add_argument("--min_df", type=float, default=0.1)

    args = parser.parse_args()

    # TODO: apply random seed to scikit-learn methods
    if args.seed == -1:
        random.seed(None)
        args.seed = random.randrange(0, int(1e4))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.verbose = not args.quiet
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    globals()[args.task](args)
