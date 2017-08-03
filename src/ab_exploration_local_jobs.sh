#!/bin/bash
set -x

# 'tfgr' is not neccessary in local as it is equivalent to tfig

for dataset in 'reuters21578' '20newsgroups' 'ohsumed20k'
do
    for vect in 'tfig' 'tfchi' 'tfrf' 'tfcw'
    do
        python ab_exploration_local.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_local.csv -l LinearSVC --params 1
        python ab_exploration_local.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_local.csv --sublinear_tf -l LinearSVC --params 1
    done

done


for dataset in 'reuters21578' '20newsgroups' 'ohsumed20k'
do
    for vect in 'tfig' 'tfchi' 'tfrf' 'tfcw'
    do
        python ab_exploration_local.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_local.csv -l LinearSVC --params 2
        python ab_exploration_local.py -d "$dataset" -v "$vect" -r ../results/alphabeta/ab_local.csv --sublinear_tf -l LinearSVC --params 2
    done

done
