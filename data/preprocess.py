from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer as Cntizer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.pipeline import Pipeline
from data.corpus import MBTI_TYPES

# TODO: add multi-output option
class CountVectorizer:
    def __init__(self, posts, types, args=None):
        self.X = posts
        lab_encoder = LabelEncoder().fit(MBTI_TYPES)
        self.y = lab_encoder.transform(types)
        self.cntizer = Pipeline([
            ('cntizer', Cntizer(
                max_features=args.max_features,
                max_df=args.max_df,
                min_df=args.min_df
            )),
            ('scaler', MaxAbsScaler()),
        ])
        self.kf = StratifiedShuffleSplit(n_splits=args.n_splits)

    def __len__(self):
        return self.kf.get_n_splits()

    def __iter__(self):
        self.iter = self.kf.split(self.X, self.y)
        return self

    def __next__(self):
        try:
            train, test = next(self.iter)
        except StopIteration:
            raise StopIteration
        X_train = self.cntizer.fit_transform(self.X[train])
        y_train = self.y[train]
        X_test = self.cntizer.transform(self.X[test])
        y_test = self.y[test]
        return (X_train, y_train), (X_test, y_test)
