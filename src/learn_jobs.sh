#!/bin/bash
set -x

learn='python supervised_weighting.py'

for dataset in 'ohsumed' #'ohsumed' #'20newsgroups' #'reuters21578'
do
    for cat in {0..4}
    do
        #$learn --dataset "$dataset" --cat "$cat" --learntf True
        #$learn --dataset "$dataset" --cat "$cat"
        pass
    done

    python classification_benchmark.py -v ../vectors -r ../results/replication_ohsumed.csv --no-randomforest --no-knn --no-logisticregression --no-multinomialnb
    #python classification_benchmark.py -d $dataset -m tfidf -r ../results/replication_ohsumed_tfidf.csv --no-randomforest --no-knn --no-logisticregression --no-multinomialnb
done
