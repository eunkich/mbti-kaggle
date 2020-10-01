#! /bin/bash

for min in $(seq 5 20)
do
    for max in $(seq 45 60)
        do
                echo min: $min max: $max
                python main.py --dataset hypertext --loader  CountVectorizer --method ensemble --min_df $min --max_df $max
        done
done
~                                                                               
~                                                                               
~                                                                               
~                                                                               
~                          
