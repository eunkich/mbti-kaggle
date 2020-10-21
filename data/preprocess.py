from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer as Cntizer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MaxAbsScaler
from sklearn.pipeline import Pipeline
from data.corpus import MBTI_TYPES
import numpy as np
from transformers import AutoTokenizer


def slicer(strType: str) -> list:
    a, b, c, d = strType
    listType = [a, b, c, d]
    return listType


class CountVectorizer:
    def __init__(self, posts, types, args=None):
        self.X = posts
        self.lab_encoder = LabelEncoder().fit(MBTI_TYPES)
        self.y = self.lab_encoder.transform(types)
        if args.binary:
            s_types = []
            for elem in types:
                s_types.append(slicer(elem))
            self.lab_encoder = OrdinalEncoder().fit(s_types)
            # define binary_encoder
            self.y = self.lab_encoder.transform(s_types)

        self.cntizer = Pipeline([
            ('cntizer', Cntizer(
                max_features=args.max_features,
                max_df=args.max_df,
                min_df=args.min_df
            )),
            ('scaler', MaxAbsScaler()),
        ])
        self.kf = StratifiedShuffleSplit(
            n_splits=args.n_splits,
            random_state=args.seed
        )
        self.input_dim = args.max_features
        self.output_dim = len(np.unique(self.y))

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


class LanguageModel:
    def __init__(self, posts, types, args=None):
        tokenizer = AutoTokenizer.from_pretrained(
            args.lm,
            use_fast=True,
        )
        tokens = tokenizer(
            posts.tolist(),
            max_length=args.max_length,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=False,
            verbose=args.verbose,
        )
        self.X = np.asarray(tokens['input_ids'])

        self.lab_encoder = LabelEncoder().fit(MBTI_TYPES)
        self.y = self.lab_encoder.transform(types)
        if args.binary:
            s_types = []
            for elem in types:
                s_types.append(slicer(elem))
            self.lab_encoder = OrdinalEncoder().fit(s_types)
            self.y = self.lab_encoder.transform(s_types)
        self.kf = StratifiedShuffleSplit(
            n_splits=args.n_splits,
            random_state=args.seed
        )
        self.input_dim = self.X.shape[1]
        self.output_dim = len(np.unique(self.y))

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
        X_train = self.X[train]
        y_train = self.y[train]
        X_test = self.X[test]
        y_test = self.y[test]
        return (X_train, y_train), (X_test, y_test)
