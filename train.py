import numpy as np
from scipy.sparse import issparse
import pandas as pd
import torch
import torch.nn as nn
import models
import os
from utils import log
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score


def sgd(loader, args):
    # K-fold validation
    args_dict = vars(args)
    result_col = ['accuracy','f1'] + list(args_dict.keys())
    result = pd.DataFrame(
        data=np.zeros((args.epochs + 1, len(result_col))),
        index=np.arange(args.epochs + 1),
        columns=result_col
    )

    result[list(args_dict.keys())] = list(args_dict.values())

    bs = args.batch_size

    log(f'Begin training model "{args.model}"', args.verbose)
    for idx, ((X_train, y_train), (X_test, y_test)) in enumerate(loader):
        log('=' * 20 + f'  Subset {idx + 1}/{len(loader)}  ' + '=' * 20)

        # Load deep learning classifier and optimizer
        clf = getattr(models, args.model)(
            loader.input_dim,
            loader.output_dim,
            args
        ).to(args.device)
        optim = torch.optim.Adam(
            clf.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=0.0
        )
        criterion = nn.CrossEntropyLoss()

        # Initial validation
        clf.eval()
        num_batch = X_test.shape[0] // bs + (X_test.shape[0] % bs != 0)
        s = 0
        accs, scores = 0, 0
        for _ in range(num_batch):
            X = X_test[s:s + bs]
            if issparse(X):
                X = X.todense()
            X = torch.FloatTensor(X).to(args.device)
            pad_size = (0, max(0, args.max_features - X.shape[1]))
            X = nn.ConstantPad1d(pad_size, 0)(X)
            y = torch.LongTensor(y_test[s:s + bs]).to(args.device)
            outputs = clf(X)
            preds = torch.max(outputs, dim=-1)[1]
            accs += (preds == y).float().mean()
            scores += f1_score(
                y.cpu().numpy(),
                preds.cpu().numpy(),
                average='weighted'
            )
            s += bs
        accs_ = accs.detach().cpu().clone().numpy()
        result.loc[0,'accuracy'] += accs_ / len(loader)
        result.loc[0,'f1'] += scores / len(loader)

        for e in range(args.epochs):
            # Training
            clf.train()
            shuffled = np.arange(X_train.shape[0])
            np.random.shuffle(shuffled)
            num_batch = X_train.shape[0] // bs + (X_train.shape[0] % bs != 0)
            s = 0
            for _ in range(num_batch):
                b_idx = shuffled[s:s + bs]
                X = X_train[b_idx]
                if issparse(X):
                    X = X.todense()
                X = torch.FloatTensor(X).to(args.device)
                pad_size = (0, max(0, args.max_features - X.shape[1]))
                X = nn.ConstantPad1d(pad_size, 0)(X)
                y = torch.LongTensor(y_train[b_idx]).to(args.device)
                outputs = clf(X)
                loss = criterion(outputs, y)

                optim.zero_grad()
                loss.backward()
                optim.step()
                s += bs

            # Validation
            clf.eval()
            num_batch = X_test.shape[0] // bs + (X_test.shape[0] % bs != 0)
            s = 0
            accs, scores = 0, 0
            for _ in range(num_batch):
                X = X_test[s:s + bs]
                if issparse(X):
                    X = X.todense()
                X = torch.FloatTensor(X).to(args.device)
                pad_size = (0, max(0, args.max_features - X.shape[1]))
                X = nn.ConstantPad1d(pad_size, 0)(X)
                y = torch.LongTensor(y_test[s:s + bs]).to(args.device)
                outputs = clf(X)
                preds = torch.max(outputs, dim=-1)[1]
                accs += (preds == y).float().mean()
                scores += f1_score(
                    y.cpu().numpy(),
                    preds.cpu().numpy(),
                    average='weighted'
                )
                s += bs

            accs /= num_batch
            scores /= num_batch
            log("Epoch {:2d} - Accuracy: {:.4f}  F1: {:.4f}".format(
                e + 1, accs, scores
            ))
            accs_ = accs.detach().cpu().clone().numpy()
            result.loc[e+1,'accuracy'] += accs_ / len(loader)
            result.loc[e+1,'f1'] += scores / len(loader)

    os.makedirs('./results', exist_ok=True)
    file_name = './results/' + "result_table.csv"


    if os.path.isfile(file_name):
        result_ = pd.read_csv(file_name, index_col=0)
        result_ = result_.append(result)
        result_.to_csv(file_name)
    else:
        result.to_csv(file_name)

    #result.to_csv('./results/' + args.output)
    log(f"{args.n_splits}-fold cross validation result:\n")
    log(f"Saved validation result to {args.output}")


def ensemble(loader, args):
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
        stack_method='predict_proba',
        verbose=args.verbose,
        n_jobs=-1,
    )

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
        C = loader.output_dim
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
        result.loc['stacking']['accuracy'] += acc
        result.loc['stacking']['f1'] += score

        log("Voting   - Accuracy: {:.4f}  F1: {:.4f}".format(acc, score))

    result /= args.n_splits

    args_dict = vars(args)
    result_col = ['accuracy', 'f1', 'weight'] + list(args_dict.keys())
    print(result_col, "\n\n\n\n\n", len(result_col), "\n\n\n\n\n")
    result_ = pd.DataFrame(
        data=np.zeros((len(clfs) + 2, len(result_col))),
        index=[c[0] for c in clfs] + ['voting', 'stacking'],
        columns=result_col
    )
    result_[list(args_dict.keys())] = list(args_dict.values())
    result_['accuracy'] = result['accuracy']
    result_['f1'] = result['f1']
    result_['weight'] = result['weight']

    os.makedirs('./results', exist_ok=True)
    file_name = './results/' + "result_table.csv"

    if os.path.isfile(file_name):
        result_ = pd.read_csv(file_name, index_col=0)
        _result_ = result_.append(result_)
        _result_.to_csv(file_name)
    else:
        result_.to_csv(file_name)
    log(f"{args.n_splits}-fold cross validation result:\n")
    print(result, end='\n\n')
    log(f"Saved validation result to {args.output}")
