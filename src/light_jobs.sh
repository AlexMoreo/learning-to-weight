#!/bin/bash
set -x

exe='python light_supervised_weighting.py'

for dataset in 'ohsumed' '20newsgroups' 'reuters21578'
do
    for run in 0
    do
    	for cat in {0..115}
        do
            log=../logs/cat"$cat".txt
            for idf in 'infogain' 'chisquare' 'gainratio' 'idf' 'rel_factor'
            do

                $exe --dataset $dataset --cat "$cat" --outdir ../vectors/"$dataset"_light --idflike $idf --run "$run"

            done
        done
    done
    python classification_benchmark.py -v ../vectors/"$dataset"_light --no-randomforest -r ../results/"$dataset"_light.csv
done
