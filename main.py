import argparse
import random
import numpy as np
import torch
from train import ensemble, sgd
from data.corpus import load_kaggle, load_kaggle_masked, load_hypertext
from data.preprocess import CountVectorizer, LanguageModel


def main():
    parser = argparse.ArgumentParser(
            description="MBTI classification from kaggle dataset"
    )
    dataset = ['load_kaggle', 'load_kaggle_masked', 'load_hypertext']
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
    parser.add_argument('-o', "--output", type=str, default='result.csv')
    # User defined string at the end of the filename of the result

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
    # Unified result filename format
    args.output = f'binary_{args.binary}_{args.dataset}_{args.loader}_{args.method}_' + args.output

    args.verbose = not args.quiet
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wrapper(args)


def binary_task(loader, args):
    original_y = loader.y
    for i in range(4):
        print('Target Category | ', loader.lab_encoder.categories_[i])
        loader.y = original_y[:,i]
        globals()[args.method](loader, args)


def wrapper(args):
    posts, types = globals()[args.dataset](args=args)
    loader = globals()[args.loader](posts, types, args)

    if not args.binary:
        globals()[args.method](loader, args)
    else:
        binary_task(loader, args)


if __name__ == '__main__':
    main()
