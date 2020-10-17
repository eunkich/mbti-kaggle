"""code modified from MBTI_preprocessing_JunHoo.ipynb
   and Multiclass and multi-output classification.ipynb
"""
import os
import requests
import pickle5 as pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from utils import log
import unidecode
import contractions

URL = "https://docs.google.com/uc?export=download"
ID_RAW = "1bphcWY5dd2J2rUgklIAYofd3hmsADdHX"
ID_PROCESSED = "1XPuBDU0lIX4rd8eQrH1Aqo0FrgKJdvIs"


MBTI_TYPES = ['infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp',
              'isfp', 'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj']
SLICED_TYPES = [['i', 'n', 'f', 'j'],
                ['e', 'n', 't', 'p'],
                ['i', 'n', 't', 'p'],
                ['i', 'n', 't', 'j'],
                ['e', 'n', 't', 'j'],
                ['e', 'n', 'f', 'j'],
                ['i', 'n', 'f', 'p'],
                ['e', 'n', 'f', 'p'],
                ['i', 's', 'f', 'p'],
                ['i', 's', 't', 'p'],
                ['i', 's', 'f', 'j'],
                ['i', 's', 't', 'j'],
                ['e', 's', 't', 'p'],
                ['e', 's', 'f', 'p'],
                ['e', 's', 't', 'j'],
                ['e', 's', 'f', 'j']]
MBTI_TOKEN = '<MBTI>'
HYPERTEXT_TOKEN = '<HYPER>'


def download(file_id, filename):
    if not os.path.isfile(filename):
        # Request file from URL
        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value}
                response = session.get(URL, params=params, stream=True)

        # Download file
        with open(filename, 'wb') as f:
            f.write(response.content)

    return pd.read_csv(filename)


def preprocess(text, lemmatize=True, remove_stop_words=True):
    # remove hypertext
    cachedStopWords = stopwords.words("english")
    lemmatiser = WordNetLemmatizer()
    temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # convert accented characters
    temp = unidecode.unidecode(temp)
    # expand contradictions
    temp = contractions.fix(temp)
    # extract only english words
    temp = re.sub("[^a-zA-Z]", " ", temp)
    # remove extra masks
    temp = re.sub("reyphyper reyphyper", ' ', temp).lower()
    # 띄어쓰기 한칸으로 맞추기
    temp = re.sub('\s+', ' ', temp).lower()
    # 어근 추출
    if lemmatize:
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
    temp = re.sub("reyphyper", '<hypertext>', temp)
    # 다시 붙이기
    return temp


def preprocess_kaggle(data, lemmatize=True, remove_stop_words=True,
                      verbose=False):
    # Remove and clean comments
    posts = []
    log("Preprocessing in kaggle-fashion", verbose)
    nltk.download('stopwords')
    nltk.download('wordnet')
    posts = data["posts"].apply(
        preprocess,
        args=[lemmatize, remove_stop_words]
    )
    # Convert to numpy array
    posts = np.array(posts)
    types = np.array(data['type'].str.lower())
    return posts, types


def load_kaggle(filename='kaggle.pkl', args=None, verbose=False, **kwargs):
    verbose = args.verbose
    lemmatize = True
    if args.loader == 'LanguageModel':
        filename = 'kaggle_nolem.pkl'
        lemmatize = False

    if not os.path.isfile(filename):
        data = download(ID_RAW, 'mbti_1.csv')
        posts, types = preprocess_kaggle(data, verbose=verbose,
                                         lemmatize=lemmatize, **kwargs)
        dict_data = {'posts': posts, 'type': types}
        with open(filename, 'wb') as handle:
            pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        log(f"Loading preprocessed data from {filename}", verbose)
        with open(filename, 'rb') as handle:
            dict_data = pickle.load(handle)
        posts = dict_data['posts']
        types = dict_data['type']

    return posts, types


def load_kaggle_masked(filename='kaggle_masked.pkl', args=None, **kwargs):
    verbose = args.verbose
    lemmatize = True
    if args.loader == 'LanguageModel':
        filename = 'kaggle_masked_nolem.pkl'
        lemmatize = False

    if not os.path.isfile(filename):
        data = download(ID_RAW, 'mbti_1.csv')
        posts, types = preprocess_kaggle(data, verbose=verbose,
                                         lemmatize=lemmatize, **kwargs)

        log("Masking MBTI types", verbose)
        for idx in range(len(posts)):
            words = posts[idx].split(" ")
            masked = [MBTI_TOKEN if w in MBTI_TYPES else w for w in words]
            posts[idx] = " ".join(masked)

        dict_data = {'posts': posts, 'type': types}
        with open(filename, 'wb') as handle:
            pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        log(f"Loading preprocessed data from {filename}", verbose)
        with open(filename, 'rb') as handle:
            dict_data = pickle.load(handle)
        posts = dict_data['posts']
        types = dict_data['type']

    return posts, types


def load_hypertext(filename='hypertext.pkl', args=None, **kwargs):
    verbose = args.verbose
    lemmatize = True
    if args.loader == 'LanguageModel':
        filename = 'hypertext_nolem.pkl'
        lemmatize = False

    if not os.path.isfile(filename):
        data = download(ID_PROCESSED, 'mbti_masked.csv')
        posts, types = preprocess_kaggle(data, verbose=verbose,
                                         lemmatize=lemmatize, **kwargs)

        log("Masking MBTI types", verbose)
        for idx in range(len(posts)):
            words = posts[idx].split(" ")
            masked = [MBTI_TOKEN if w in MBTI_TYPES else w for w in words]
            posts[idx] = " ".join(masked)

        dict_data = {'posts': posts, 'type': types}
        with open(filename, 'wb') as handle:
            pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        log(f"Loading preprocessed data from {filename}", verbose)
        with open(filename, 'rb') as handle:
            dict_data = pickle.load(handle)
        posts = dict_data['posts']
        types = dict_data['type']

    return posts, types
