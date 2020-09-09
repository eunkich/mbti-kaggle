import argparse
import random
import numpy as np
import torch
from train import ensemble, sgd


def kaggle(args):
    """Reproduce the original preprocessing from kaggle"""
    from data.corpus import load_kaggle
    from data.preprocess import CountVectorizer

    posts, types = load_kaggle(verbose=args.verbose)
    loader = CountVectorizer(posts, types, args)
    ensemble(loader, args)


def kaggle_masked(args):
    """Reproduce + mask MBTI types"""
    from data.corpus import load_kaggle_masked
    from data.preprocess import CountVectorizer

    posts, types = load_kaggle_masked(verbose=args.verbose)
    loader = CountVectorizer(posts, types, args)
    ensemble(loader, args)


def kaggle_hypertext(args):
    """Replace hypertext with its contents"""
    from data.corpus import load_hypertext
    from data.preprocess import CountVectorizer

    posts, types = load_hypertext(verbose=args.verbose)
    loader = CountVectorizer(posts, types, args)
    ensemble(loader, args)


def kaggle_sgd(args):
    from data.corpus import load_kaggle_masked
    from data.preprocess import CountVectorizer

    posts, types = load_kaggle_masked(verbose=args.verbose)
    loader = CountVectorizer(posts, types, args)
    sgd(loader, args)


def lm_sgd(args):
    from data.corpus import load_kaggle_masked
    from data.preprocess import LanguageModel

    posts, types = load_kaggle_masked(
        filename='kaggle_nolem.pkl',
        lemmatize=False,
        verbose=args.verbose,
    )
    loader = LanguageModel(posts, types, args,
                           filename='kaggle_nolem_embed.npy')
    sgd(loader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MBTI classification from kaggle dataset"
    )
    tasks = ['kaggle', 'kaggle_masked', 'kaggle_hypertext']
    tasks += ['kaggle_sgd', 'lm_sgd']
    parser.add_argument("--task", type=str, choices=tasks)
    parser.add_argument("--seed", type=str, default=-1)
    parser.add_argument('-q', "--quiet", action="store_true")
    parser.add_argument('-o', "--output", type=str, default='result.csv')

    parser.add_argument_group("Preprocessing options")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--max_features", type=int, default=1500)
    parser.add_argument("--max_df", type=float, default=0.5)
    parser.add_argument("--min_df", type=float, default=0.1)
    parser.add_argument("--lm", type=str, default='bert-base-uncased')
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument_group("Model options")
    parser.add_argument("--model", type=str, default='mlp3')
    parser.add_argument("--bn", action='store_true')
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=2048)

    parser.add_argument_group("Training options")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)

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
