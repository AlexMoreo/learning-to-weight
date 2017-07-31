#!/bin/bash
set -x

for dataset in 'reuters21578' '20newsgroups' 'ohsumed20k'
do
    for vect in 'tfidf' 'bm25' 'tfig' 'tfgr' 'tfchi' 'tfrf' 'tfcw'
    do
        python ab_plotter.py -d "$dataset" -v "$vect" -r .../plots/AB_plots --sublinear_tf -l LinearSVC
        python ab_plotter.py -d "$dataset" -v "$vect" -r .../plots/AB_plots -l LinearSVC
    done

done
