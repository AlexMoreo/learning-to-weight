#!/bin/bash
set -x

exe='python classification_benchmark.py'
result='-r ../results/baselines_svm.csv'
classification='--classification binary'

for dataset in 'reuters21578' #'ohsumed' '20newsgroups'
do
    $exe -d $dataset $result -m binary --fs 0.1 $classification
    $exe -d $dataset $result -m tf --fs 0.1 $classification
    $exe -d $dataset $result -m tf --fs 0.1 $classification --sublinear_tf
    $exe -d $dataset $result -m tfidf --fs 0.1 $classification
    $exe -d $dataset $result -m tfidf --fs 0.1 $classification --sublinear_tf
    $exe -d $dataset $result -m bm25 --fs 0.1 $classification
    $exe -d $dataset $result -m tfig --fs 0.1 $classification
    $exe -d $dataset $result -m tfig --fs 0.1 $classification --sublinear_tf
    $exe -d $dataset $result -m tfchi2 --fs 0.1 $classification
    $exe -d $dataset $result -m tfchi2 --fs 0.1 $classification --sublinear_tf
    $exe -d $dataset $result -m tfcw --fs 0.1 $classification
    $exe -d $dataset $result -m tfcw --fs 0.1 $classification --sublinear_tf
    $exe -d $dataset $result -m tfrf --fs 0.1 $classification
    $exe -d $dataset $result -m tfrf --fs 0.1 $classification --sublinear_tf
done


