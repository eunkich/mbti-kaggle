import numpy as np
import pandas as pd
import models
from data.corpus import MBTI_TYPES
from utils import log, one_hot
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score


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