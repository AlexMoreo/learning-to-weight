#!/bin/bash
set -x

exe='python simple_supervised_weighting.py'
learn='python multi_supervised_weighting.py'

for dataset in 'ohsumed' '20newsgroups' 'reuters21578'
do
    for cat in {0..115}
    do
        #$exe --dataset "$dataset" --cat "$cat" --fs 0.1
        $learn --dataset "$dataset" --cat "$cat" --fs 0.1
    done

    python classification_benchmark.py -v ../vectors -r ../results/simple_learn_svm.csv
done
