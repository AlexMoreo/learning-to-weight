#!/bin/bash
set -x

for dataset in 'reuters21578' '20newsgroups' 'ohsumed20k'
do
    for vect in 'tfidf' 'bm25' 'tfig' 'tfgr' 'tfchi' 'tfrf' 'tfcw'
    do
        python ab_exploration.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_table.csv --sublinear_tf -l LinearSVC --not_explore_ab
        python ab_exploration.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_table.csv -l LinearSVC --not_explore_ab
        python ab_exploration.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_table.csv --sublinear_tf -l LinearSVC
        python ab_exploration.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_table.csv -l LinearSVC
    done

done
