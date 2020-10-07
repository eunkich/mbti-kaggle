#! /bin/bash
dataset=('kaggle' 'kaggle_masked' 'hypertext')
loader=('CountVectorizer' 'LanguageModel')
method=('ensemble' 'sgd')
for dat_ in $(seq 0 2)
do
    for max in $(seq 45 60)
        do
                for met_ in $(seq 0 1)
                do
                    echo python main.py --dataset ${dataset[$dat_]} --loader  ${loader[$loa_]} --method ${method[$met_]} 
                    #python main.py --dataset ${dataset[$dat_]} --loader  ${loader[$loa_]} --method ${method[$met_]} 
                done
        done
done