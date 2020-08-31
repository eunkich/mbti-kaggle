"""code modified from MBTI_preprocessing_JunHoo.ipynb
   and Multiclass and multi-output classification.ipynb
"""
import os
import requests
import pickle
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from contextlib import redirect_stdout
from tqdm import tqdm
from utils import log

ID_RAW = "1bphcWY5dd2J2rUgklIAYofd3hmsADdHX"
ID_PROCESSED = "1-0yxLrIpq6f_avR3OfRvVVVtkVMKJ_Ic"
URL = "https://docs.google.com/uc?export=download"

MBTI_TYPES = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
              'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']


def download(file_id, filename):
    if not os.path.isfile(filename):
        # Request file from URL
        session = requests.Session()
        response = session.get(URL, params = {'id': file_id}, stream = True)
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value}
                response = session.get(URL, params=params, stream=True)
    
        # Download file
        with open(filename, 'wb') as f:
            f.write(response.content) 

    return pd.read_csv(filename)


def load_kaggle(filename='kaggle.pkl', remove_stop_words=True, verbose=False):
    if not os.path.isfile(filename):
        data = download(ID_RAW, 'mbti_1.csv')
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
            if remove_stop_words:
                temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
            else:
                temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
            posts.append(temp)
    
        # Convert to numpy array
        posts = np.array(posts)
        types = np.array(data['type'])
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

# TODO: add other preprocessing methods
#       e.g. load junhoo's dataset
