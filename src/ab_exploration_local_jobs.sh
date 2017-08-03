#!/bin/bash
set -x

#this experiment do not explore the parameter C of the svm. Instead it performs 5-fold cross-validation (the previous was 3-fold cv)

for dataset in 'ohsumed20k' '20newsgroups' 'reuters21578'
do
    for vect in 'tfgss' 'tfpmi' 'tfigpos' #tfcw' 'tfrf' 'tfchi' 'tfgr' 'tfig'
    do
        python ab_exploration_local.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_local.csv -l LinearSVC --params 0
        python ab_exploration_local.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_local.csv --sublinear_tf -l LinearSVC --params 0
        python ab_exploration_local.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_local.csv -l LinearSVC --params 1
        python ab_exploration_local.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_local.csv --sublinear_tf -l LinearSVC --params 1
    done

    #python ab_exploration_local.py -d "$dataset" -v tfidf -r ../results/alphabeta/ab_local.csv -l LinearSVC --params 0
    #python ab_exploration_local.py -d "$dataset" -v tfidf -r ../results/alphabeta/ab_local.csv -l LinearSVC --params 0 --sublinear_tf
    #python ab_exploration_local.py -d "$dataset" -v bm25 -r ../results/alphabeta/ab_local.csv -l LinearSVC --params 0
    #python ab_exploration_local.py -d "$dataset" -v tfidf -r ../results/alphabeta/ab_local.csv -l LinearSVC --params 1
    #python ab_exploration_local.py -d "$dataset" -v tfidf -r ../results/alphabeta/ab_local.csv -l LinearSVC --params 1 --sublinear_tf
    #python ab_exploration_local.py -d "$dataset" -v bm25 -r ../results/alphabeta/ab_local.csv -l LinearSVC --params 1

done

