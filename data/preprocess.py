import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer as Cntizer
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from data.corpus import MBTI_TYPES
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from utils import log, restart
from tqdm import tqdm


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
    def __init__(self, posts, types, args=None, filename='embed.npy'):
        self.args = args
        # FIXME: multiprocessing lock issue in tokenizer
        if not os.path.isfile(filename):
            self.create_embedding(posts, filename)
            log("Restarting process due to multiprocessing fork issues")
            restart()

        log(f"Loading embedding from {filename}", args.verbose)
        self.X = np.load(filename, mmap_mode='r')

        lab_encoder = LabelEncoder().fit(MBTI_TYPES)
        self.y = lab_encoder.transform(types)
        self.scaler = MinMaxScaler()
        self.kf = StratifiedShuffleSplit(n_splits=args.n_splits)
        self.input_dim = self.X.shape[1]
        self.output_dim = len(np.unique(self.y))

    def create_embedding(self, posts, filename):
        # TODO: allow arbitrary length
        log(f'Embedding data using pretrained model "{self.args.lm}"',
            self.args.verbose)
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.lm,
            use_fast=True
        )
        tokens = tokenizer(
            posts.tolist(),
            max_length=self.args.max_length,
            truncation=True,
            padding='max_length',
            verbose=self.args.verbose,
            return_tensors='pt'
        )
        device = self.args.device
        model = AutoModel.from_pretrained(self.args.lm).to(device)

        bs = self.args.batch_size
        idx_range = range(len(posts) // bs)
        if self.args.verbose:
            idx_range = tqdm(idx_range)

        hidden_states = []
        for idx in idx_range:
            s = idx * bs
            e = (idx + 1) * bs
            with torch.no_grad():
                hidden_state = model(
                    input_ids=tokens['input_ids'][s:e].to(device),
                    token_type_ids=tokens['token_type_ids'][s:e].to(device),
                    attention_mask=tokens['attention_mask'][s:e].to(device),
                    return_dict=True
                ).last_hidden_state
                hidden_state = hidden_state.view(e - s, -1).cpu().numpy()
            hidden_states.append(hidden_state)

        if len(posts) % bs != 0:
            s = (idx + 1) * bs
            e = len(posts)
            with torch.no_grad():
                hidden_state = model(
                    input_ids=tokens['input_ids'][s:e].to(device),
                    token_type_ids=tokens['token_type_ids'][s:e].to(device),
                    attention_mask=tokens['attention_mask'][s:e].to(device),
                    return_dict=True
                ).last_hidden_state
                hidden_state = hidden_state.view(e - s, -1).cpu().numpy()
            hidden_states.append(hidden_state)

        X = np.concatenate(hidden_states, axis=0)
        np.save(filename, X)

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
        X_train = self.scaler.fit_transform(self.X[train])
        y_train = self.y[train]
        X_test = self.scaler.transform(self.X[test])
        y_test = self.y[test]
        return (X_train, y_train), (X_test, y_test)
