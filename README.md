# mbti-kaggle

Our take on the [MBTI dataset uploaded on Kaggle](https://www.kaggle.com/datasnaek/mbti-type),  
using the skillset learned from the online version of [Stanford's CS224n](http://web.stanford.edu/class/cs224n/).


## Setup

This repository was run in Python 3.8.  
Dependencies can be installed via pip:
```python
pip install -r requirements.txt
```


## Experimental Results

### Multiclass Classification

Classification accuracy and F1 score under 3-fold cross validation (single seed)

| Preprocessing | Vectorization | Classifier | Accuracy | F1 | 
|---|---|---|---|---|
| Original  | CountVectorizer    | Classical ML | 0.6778        | 0.6665        |
| Original  | CountVectorizer    | MLP          | 0.6016        | 0.5747        |
| Original  | LanguageModel      | MLP          | **0.7796**    | **0.7771**    |
| Masked    | CountVectorizer    | Classical ML | 0.4854        | 0.4476        |
| Masked    | CountVectorizer    | MLP          | 0.4360        | 0.4058        |
| Masked    | LanguageModel      | MLP          | **0.5530**    | **0.5420**    |
| Hypertext | CountVectorizer    | Classical ML | 0.4889        | 0.4508        |
| Hypertext | CountVectorizer    | MLP          | 0.4432        | 0.4107        |
| Hypertext | LanguageModel      | MLP          | **0.5534**    | **0.5441**    |


### Binary Classification


## Usage

The commands for reproducing the results for multiclass classification are shown below.

Original + CountVectorizer + Classical ML
```python
python main.py --dataset kaggle \
               --loader CountVectorizer \
               --method ensemble \
               --n_splits 3 \
               --seed 100
```

Masked + CountVectorizer + MLP
```python
python main.py --dataset kaggle_masked \
               --loader CountVectorizer \
               --method sgd \
               --model mlp3 \
               --batch_size 16 \
               --lr 2e-5 \
               --epochs 10 \
               --dropout 0.1 \
               --bn \
               --n_splits 3 \
               --seed 100
```

Hypertext + LanguageModel + MLP  
Note that the required vram is about 42Gb, due to the length of the input sequence.
```python
python main.py --dataset hypertext \
               --loader LanguageModel \
               --method sgd \
               --model lm_classifier \
               --lm xlnet-base-cased \
               --max_length 1500 \
               --batch_size 4 \
               --lr 2e-5 \
               --epochs 5 \
               --n_splits 3 \
               --seed 100
```

