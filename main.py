import os
import sys
import argparse
import random
import numpy as np
import torch
from utils import log


def test(args):
    import models
    from utils import one_hot
    from data.corpus import load_kaggle, MBTI_TYPES
    from data.preprocess import CountVectorizer
    from sklearn.ensemble import StackingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score
    import pandas as pd

    # Load models and ensemble
    clfs = []
    for m in dir(models):
        try:
            clf = getattr(models, m)(args)
            if not hasattr(clf, 'predict_proba'):
                clf = CalibratedClassifierCV(clf)
            clfs.append((m, clf))
        except TypeError:
            pass
        except Exception as e:
            raise e
    stack_clf = StackingClassifier(
        estimators=clfs,
        final_estimator=models.logistic(args),
        verbose=args.verbose,
        n_jobs=-1,
    )

    # Load data
    X, y = load_kaggle(verbose=args.verbose)
    loader = CountVectorizer(X, y, args)

    # K-fold validation
    result = pd.DataFrame(
        data=np.zeros((len(clfs) + 2, 3)),
        index=[c[0] for c in clfs] + ['voting', 'stacking'],
        columns=['accuracy', 'f1', 'weight']
    )
    log('Begin training {} classifiers: {}'.format(
        len(clfs),
        " ".join([c[0] for c in clfs])
    ), args.verbose)
    for idx, ((X_train, y_train), (X_test, y_test)) in enumerate(loader):
        log('=' * 20 + f'  Subset {idx + 1}/{len(loader)}  ' + '=' * 20)

        # Train classifiers
        stack_clf.fit(X_train, y_train)
        pred = stack_clf.predict(X_test)
        acc = (pred == y_test).mean()
        score = f1_score(y_test, pred, average='weighted')
        result.loc['stacking']['accuracy'] += acc
        result.loc['stacking']['f1'] += score
        log("Stacking - Accuracy: {:.4f}  F1: {:.4f}".format(acc, score))

        # Record statistics for each classifier
        C = len(MBTI_TYPES)
        probs = np.zeros((len(y_test), C))
        for idx, clf in enumerate(stack_clf.estimators_):
            pred = clf.predict(X_test)
            acc = (pred == y_test).mean()
            score = f1_score(y_test, pred, average='weighted')
            result_clf = result.iloc[idx]
            result_clf['accuracy'] += acc
            result_clf['f1'] += score

            probs += clf.predict_proba(X_test)
            coef = stack_clf.final_estimator_.coef_[:, C * idx:(C + 1) * idx]
            result_clf['weight'] += np.linalg.norm(coef)

        # Record statistics for voting
        pred = np.argmax(probs, axis=1)
        acc = (pred == y_test).mean()
        score = f1_score(y_test, pred, average='weighted')
        result.loc['voting']['accuracy'] += acc
        result.loc['voting']['f1'] += score
        log("Voting   - Accuracy: {:.4f}  F1: {:.4f}".format(acc, score))

    result /= args.n_splits
    result.to_csv(args.output)
    log(f"{args.n_splits}-fold cross validation result:\n")
    print(result, end='\n\n')
    log(f"Saved validation result to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MBTI classification from kaggle dataset"
    )
    parser.add_argument("--task", type=str, default='test')
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
