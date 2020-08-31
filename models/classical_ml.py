from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import xgboost as xgb
from data.corpus import MBTI_TYPES

__all__ = ['pass_agg', 'linear_sgd', 'logistic']    # Linear models
__all__ += ['svc_linear']                           # Support Vector Machines
__all__ += ['naive_bayes']                          # Naive Bayes
__all__ += ['xgb_class', 'random_forest']           # Decision trees
__all__ += ['knn']                                  # K-nearest neighbor
__all__ += ['mlp']                                  # Neural Network


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def gp_rbf(args):
    return Pipeline([
        ('to_dense', DenseTransformer()),
        ('gaussian_process', GaussianProcessClassifier(
            kernel=RBF(1.0),
            max_iter_predict=100,
        ))
    ])


def random_forest(args):
    return Pipeline([
        ('to_dense', DenseTransformer()),
        ('random_forest', RandomForestClassifier(n_estimators=300))
    ])


def mlp(args):
    return MLPClassifier(max_iter=10000)


def knn(args):
    return KNeighborsClassifier()


def svc_linear(args):
    return LinearSVC(max_iter=10000)

# FIXME: too slow - add svm-gpu implementation?
def svc_rbf(args):
    return SVC(kernel='rbf', max_iter=10000)


def naive_bayes(args):
    return MultinomialNB()


def pass_agg(args):
    return PassiveAggressiveClassifier(max_iter=10000)


def linear_sgd(args):
    return SGDClassifier(
        loss='hinge',
        penalty='l1',
        alpha=1e-2,
        max_iter=10000,
    )


def logistic(args):
    return LogisticRegression(
        multi_class='multinomial',
        max_iter=10000,
    )


def xgb_class(args):
    return xgb.XGBClassifier(
        objective='multi:softprob',
        eta=0.6,
        n_estimators=300,
        subsample=0.93,
        max_depth=2,
        verbosity=0,
        n_jobs=8,
        num_class=len(MBTI_TYPES),  # FIXME: add as argument
        gpu_id=0,
        tree_method='gpu_hist'
    )
