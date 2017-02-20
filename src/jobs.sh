#!/bin/bash
set -x

exe='python classification_benchmark.py'
dataset ='-d reuters21578'
result = '-r ../results/baselines_reuters_svm_multiclass.csv'

$exe $dataset $result -m binary --fs 0.1 --classification multiclass
$exe $dataset $result -m tf --fs 0.1 --classification multiclass
$exe $dataset $result -m tf --fs 0.1 --classification multiclass --sublinear_tf
$exe $dataset $result -m tfidf --fs 0.1 --classification multiclass
$exe $dataset $result -m tfidf --fs 0.1 --classification multiclass --sublinear_tf
$exe $dataset $result -m bm25 --fs 0.1 --classification multiclass
$exe $dataset $result -m tfig --fs 0.1 --classification multiclass
$exe $dataset $result -m tfig --fs 0.1 --classification multiclass --sublinear_tf
$exe $dataset $result -m tfchi2 --fs 0.1 --classification multiclass
$exe $dataset $result -m tfchi2 --fs 0.1 --classification multiclass --sublinear_tf
$exe $dataset $result -m tfcw --fs 0.1 --classification multiclass
$exe $dataset $result -m tfcw --fs 0.1 --classification multiclass --sublinear_tf
$exe $dataset $result -m tfrf --fs 0.1 --classification multiclass
$exe $dataset $result -m tfrf --fs 0.1 --classification multiclass --sublinear_tf
$exe $dataset $result -m tfgr --fs 0.1 --classification multiclass --global_policy max
$exe $dataset $result -m tfgr --fs 0.1 --classification multiclass --sublinear_tf --global_policy max
$exe $dataset $result -m tfgr --fs 0.1 --classification multiclass --global_policy sum
$exe $dataset $result -m tfgr --fs 0.1 --classification multiclass --sublinear_tf --global_policy sum
$exe $dataset $result -m tfgr --fs 0.1 --classification multiclass --global_policy ave
$exe $dataset $result -m tfgr --fs 0.1 --classification multiclass --sublinear_tf --global_policy ave
$exe $dataset $result -m tfgr --fs 0.1 --classification multiclass --global_policy wave
$exe $dataset $result -m tfgr --fs 0.1 --classification multiclass --sublinear_tf --global_policy wave


