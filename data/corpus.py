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
from tqdm import tqdm
from utils import log

URL = "https://docs.google.com/uc?export=download"
ID_RAW = "1bphcWY5dd2J2rUgklIAYofd3hmsADdHX"
ID_PROCESSED = "1-0yxLrIpq6f_avR3OfRvVVVtkVMKJ_Ic"

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


def preprocess_kaggle(data, lemmatize=True, remove_stop_words=True,
                      verbose=False):
    nltk.download('stopwords', quiet=not verbose)
    nltk.download('wordnet', quiet=not verbose)
    cachedStopWords = stopwords.words("english")
    lemmatiser = WordNetLemmatizer()

    # Remove and clean comments
    posts = []
    rows = data.iterrows()
    log("Preprocessing in kaggle-fashion", verbose)
    if verbose:
        rows = tqdm(rows, total=len(data))
    for row in rows:
        raw = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', raw)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub('\s+', ' ', temp).lower()
        if lemmatize:
            if remove_stop_words:
                temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
            else:
                temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
        posts.append(temp)

    # Convert to numpy array
    posts = np.array(posts)
    types = np.array(data['type'].str.lower())
    return posts, types


def load_kaggle(filename='kaggle.pkl', args=None, verbose=False, **kwargs):
    verbose = args.verbose
    if args.loader == 'LanguageModel':
        filename = 'kaggle_nolem.pkl'
        lemmatize = False
    
    if not os.path.isfile(filename):
        data = download(ID_RAW, 'mbti_1.csv')
        posts, types = preprocess_kaggle(data, verbose=verbose, **kwargs)
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
    if args.loader == 'LanguageModel':
        filename = args.dataset + args.model + 'kaggle_masked_nolem.pkl'
        lemmatize = False
    
    if not os.path.isfile(filename):
        data = download(ID_RAW, 'mbti_1.csv')
        posts, types = preprocess_kaggle(data, verbose=verbose, **kwargs)

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


def load_hypertext(filename='hypertext.pkl', args=None, remove_stop_words=True,
                   verbose=False):
    verbose = args.verbose
    if args.loader == 'LanguageModel':
        filename = 'hypertext_nolem.pkl'
        lemmatize = False
    
    
    if not os.path.isfile(filename):
        data = download(ID_PROCESSED, 'mbti_preprocessed.csv')

        nltk.download('stopwords', quiet=not verbose)
        nltk.download('wordnet', quiet=not verbose)
        cachedStopWords = stopwords.words("english")
        lemmatiser = WordNetLemmatizer()

        # Remove and clean comments
        posts = []
        rows = data['preprocessed']
        log("Preprocessing in kaggle-fashion", verbose)
        if verbose:
            rows = tqdm(rows, total=len(data))
        for row in rows:
            temp = re.sub('\s+', ' ', row).lower()
            if remove_stop_words:
                temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
            else:
                temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
            posts.append(temp)

        # Convert to numpy array
        posts = np.array(posts)
        types = np.array(data['type'].str.lower())
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
