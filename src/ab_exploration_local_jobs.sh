#!/bin/bash
set -x

#this experiment do not explore the parameter C of the svm. Instead it performs 5-fold cross-validation (the previous was 3-fold cv)

for dataset in 'reuters21578' '20newsgroups' 'ohsumed20k'
do
    for vect in 'tfig' 'tfgr' 'tfchi' 'tfrf' 'tfcw'
    do
        python ab_exploration_local.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_local.csv -l LinearSVC --params 1
        python ab_exploration_local.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_local.csv --sublinear_tf -l LinearSVC --params 1
    done

done


for dataset in 'reuters21578' '20newsgroups' 'ohsumed20k'
do
    for vect in 'tfig' 'tfgr' 'tfchi' 'tfrf' 'tfcw'
    do
        python ab_exploration_local.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_local.csv -l LinearSVC --params 2
        python ab_exploration_local.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_local.csv --sublinear_tf -l LinearSVC --params 2
    done

done
