#!/bin/bash
set -x

exe='python supervised_weighting.py'
mplot='--plotmode img'
batch='--batchsize 1000'

for dataset in 'imdb'
do
    dresult='--resultcontainer ../results/newresults.csv'
    for run in {0..4}
    do
    #	for cat in {0..23}
        for cat in 1
        do
            log=../logs/cat"$cat".txt
            dplot=../plot/"$dataset"/C"$cat"
            $exe --dataset $dataset --cat "$cat" --normalize --forcepos --linidf $batch $dresult $mplot --plotdir "$dplot"/norm_pos --outdir ../vectors/"$dataset" --run "$run" |& tee -a $log
            $exe --dataset $dataset --cat "$cat" --normalize --noforcepos --linidf $batch $dresult $mplot --plotdir "$dplot"/norm_nopos --outdir ../vectors/"$dataset" --run "$run" |& tee -a $log
            $exe --dataset $dataset --cat "$cat" --normalize --forcepos --nolinidf $batch $dresult $mplot --plotdir "$dplot"/norm_pos_nolin --outdir ../vectors/"$dataset" --run "$run" |& tee -a $log
            $exe --dataset $dataset --cat "$cat" --normalize --noforcepos --nolinidf $batch $dresult $mplot --plotdir "$dplot"/norm_nopos_nolin --outdir ../vectors/"$dataset" --run "$run" |& tee -a $log


            #$exe --dataset $dataset --cat "$cat" --nonormalize --pretrain off --forcepos $batch $dresult $mplot --plotdir "$dplot"/nonorm_pos --outdir ../vectors/"$dataset" --run "$run" |& tee -a $log
            #$exe --dataset $dataset --cat "$cat" --normalize --pretrain off --noforcepos $batch $dresult $mplot --plotdir "$dplot"/norm_posneg --outdir ../vectors/"$dataset" --run "$run" |& tee -a $log
            #$exe --dataset $dataset --cat "$cat" --nonormalize --pretrain off --noforcepos $batch $dresult $mplot --plotdir "$dplot"/nonorm_posneg --outdir ../vectors/"$dataset" --run "$run" |& tee -a $log
            #$exe --dataset $dataset --cat "$cat" --normalize --pretrain infogain --forcepos $batch $dresult $mplot --plotdir "$dplot"/norm_pretraininfogain_pos --outdir ../vectors/"$dataset" --run "$run" |& tee -a $log
        done

    done

done

python classification_benchmark.py -v ../vectors/"$dataset" --no-randomforest --no-multinomialnb --no-randomforest -r ../results/"$dataset"_at01.csv




#pretrain [off, infogain, chisquare, gss]


#usage: supervised_weighting.py [-h] [--fs FS] [--cat CAT]
#                               [--batchsize BATCHSIZE] [--hidden HIDDEN]
#                               [--lrate LRATE] [--optimizer OPTIMIZER]
#                               [--normalize [NORMALIZE]] [--nonormalize]
#                               [--checkpointdir CHECKPOINTDIR]
#                               [--summariesdir SUMMARIESDIR]
#                               [--pretrain PRETRAIN] [--debug [DEBUG]]
#                               [--nodebug] [--forcepos [FORCEPOS]]
#                               [--noforcepos] [--plotmode PLOTMODE]
#                               [--plotdir PLOTDIR] [--outdir OUTDIR]
#                               [--outname OUTNAME] [--run RUN] [--notes NOTES]
#                               [--resultcontainer RESULTCONTAINER]
#                               [--maxsteps MAXSTEPS]
