#!/bin/bash
set -x

exe='python simple_supervised_weighting.py '

for dataset in 'ohsumed' '20newsgroups' 'reuters21578'
do
    for cat in {0..115}
    do
        $exe -v ../vectors -r ../results/simple_learn_svm.csv -d $dataset --cat "$cat"
    done

    python classification_benchmark.py -v ../vectors -r ../results/simple_learn_svm.csv
done
