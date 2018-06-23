#!/bin/bash
set -x

learn='python2 supervised_weighting.py'


for  run in 0
do

for dataset in 'reuters21578' # '20newsgroups' 'ohsumed'
do
    for cat in {0..115}
    do
        $learn --dataset "$dataset" --cat "$cat" --learntf True --forcepos False --run $run --outdir "../vectors_tf"
        $learn --dataset "$dataset" --cat "$cat" --learntf True --forcepos True --run $run --outdir "../vectors_tf"
        #pass
    done

done

done

python2 classification_benchmark.py -v ../vectors_tf -r ../results/reuters21578_tf.csv
