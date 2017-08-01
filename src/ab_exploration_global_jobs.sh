#!/bin/bash
set -x

#this experiment do not explore the parameter C of the svm. Instead it performs 5-fold cross-validation (the previous was 3-fold cv)

for dataset in 'reuters21578' '20newsgroups' 'ohsumed20k'
do
    for vect in 'tfidf' 'tfig' 'tfgr' 'tfchi' 'tfrf' 'tfcw'
    do
        python ab_exploration_global.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_table_noc.csv --sublinear_tf -l LinearSVC --not_explore_ab
        python ab_exploration_global.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_table_noc.csv -l LinearSVC --not_explore_ab
        python ab_exploration_global.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_table_noc.csv --sublinear_tf -l LinearSVC
        python ab_exploration_global.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_table_noc.csv -l LinearSVC
    done

   python ab_exploration_global.py -d "$dataset" -v bm25 -r ../results/alphabeta/ab_table_noc.csv -l LinearSVC --not_explore_ab
   python ab_exploration_global.py -d "$dataset" -v bm25 -r ../results/alphabeta/ab_table_noc.csv -l LinearSVC

done
