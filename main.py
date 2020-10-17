import argparse
import random
import numpy as np
import torch
import string

import train
import data.corpus as corpus
import data.preprocess as preprocess


def main():
    parser = argparse.ArgumentParser(
            description="MBTI classification from kaggle dataset"
    )
    dataset = ['kaggle', 'kaggle_masked', 'hypertext']
    loader = ['CountVectorizer', 'LanguageModel']
    method = ['ensemble', 'sgd']

    parser.add_argument("--binary", action='store_true')
    parser.add_argument("--dataset", type=str, required=True,
                        choices=dataset)
    parser.add_argument("--loader", type=str, required=True,
                        choices=loader)
    parser.add_argument("--method", type=str, required=True,
                        choices=method)

    parser.add_argument("--seed", type=str, default=-1)
    parser.add_argument('-q', "--quiet", action="store_true")

    # User defined string at the end of the filename of the result
    parser.add_argument('-o', "--output", type=str, default='result.csv')

    parser.add_argument_group("Preprocessing options")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--max_features", type=int, default=1500)
    parser.add_argument("--max_df", type=float, default=0.57)
    parser.add_argument("--min_df", type=float, default=0.09)
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

    # Unified result filename format
    table = str.maketrans('', '', string.ascii_lowercase)
    prefix = f'{args.dataset}_{args.loader.translate(table)}_{args.method}_'
    args.output = './results/' + prefix + args.output

    args.dataset = 'load_' + args.dataset
    args.verbose = not args.quiet
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wrapper(args)


def binary_task(loader, args):
    original_y = loader.y
    original_output = args.output
    for i in range(4):
        trg = loader.lab_encoder.categories_[i]
        trg = trg.tolist()
        print('Target Category | ', trg)
        loader.y = original_y[:, i]
        trg = ''.join(trg).swapcase()
        args.output = f'{trg}_{original_output}'
        getattr(train, args.method)(loader, args)


def wrapper(args):
    posts, types = getattr(corpus, args.dataset)(args=args)
    loader = getattr(preprocess, args.loader)(posts, types, args)
    if not args.binary:
        getattr(train, args.method)(loader, args)
    else:
        binary_task(loader, args)


if __name__ == '__main__':
    main()
