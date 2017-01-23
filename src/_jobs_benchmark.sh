#!/bin/bash

algconf="--no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest"


#python classification_benchmark.py -v ../vectors/reuters21578/R0 $algconf -r ../results/reuters21578_local_knn.csv &
#python classification_benchmark.py -v ../vectors/reuters21578_global/norm_pos_R0_9/R0 $algconf -r ../results/reuters21578_global_normpos_knn.csv &
#python classification_benchmark.py -v ../vectors/reuters21578_global/norm_nopos_wrongnames_R0_9/R9 $algconf -r ../results/reuters21578_global_norm_nopos_knn.csv &

#echo "waiting..."
#wait
#python classification_benchmark.py -v ../vectors/ohsumed/R0 --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -r ../results/ohsumed_local_knn.csv &
#python classification_benchmark.py -v ../vectors/ohsumed_global/norm_pos_R0_9/R0 $algconf -r ../results/ohsumed_global_normpos_knn.csv 
#python classification_benchmark.py -v ../vectors/ohsumed_global/norm_nopos_wrongnames_R0_9/R0 $algconf -r ../results/ohsumed_global_norm_nopos_knn.csv 

#python classification_benchmark.py -v ../vectors/20newsgroups/R0 $algconf -r ../results/20newsgroups_local_knn.csv 
#python classification_benchmark.py -v ../vectors/20newsgroups_global/norm_pos_R0_9/R0 $algconf -r ../results/20newsgroups_global_normpos_knn.csv 
#python classification_benchmark.py -v ../vectors/20newsgroups_global/norm_nopos_wrongnames_R3_9/R9 $algconf -r ../results/20newsgroups_global_norm_nopos_knn.csv 

#echo "last waiting... "
#wait

#python classification_benchmark.py -d 20newsgroups --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -r ../results/20newsgroups_bench_knn.csv &
#python classification_benchmark.py -d ohsumed --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -r ../results/ohsumed_bench_knn_fromhashing.csv &
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -r ../results/reuters21578_bench_knn.csv 

#wait

#python rf_classification_benchmark.py -d 20newsgroups -r ../results/20newsgroups_bench_knn_rf.csv &
#python rf_classification_benchmark.py -d ohsumed -r ../results/ohsumed_bench_knn_rf.csv &

#wait 
#python rf_classification_benchmark.py -d reuters21578 -r ../results/reuters21578_bench_knn_rf.csv


algconf="--no-linearsvm --no-multinomialnb --no-logisticregression --no-knn"
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-knn -r ../results/reuters21578_bench_randomforest.csv 
#python classification_benchmark.py -v ../vectors/reuters21578/R0 $algconf -r ../results/reuters21578_local_randomforest_R0.csv 
#python classification_benchmark.py -v ../vectors/reuters21578_global/norm_pos_R0_9/R0 $algconf -r ../results/reuters21578_global_normpos_randomforest_R0.csv 
#python classification_benchmark.py -v ../vectors/reuters21578_global/norm_nopos_wrongnames_R0_9/R9 $algconf -r ../results/reuters21578_global_norm_nopos_randomforest_R0.csv 



#python classification_benchmark.py -d ohsumed --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -r ../results/ohsumed_bench_knn_hashing.csv


algconf="--no-linearsvm --no-multinomialnb --no-logisticregression --no-knn"
#python classification_benchmark.py -v ../vectors/20newsgroups_global/norm_nopos_wrongnames_R3_9/R9 $algconf -r ../results/20newsgroups_global_norm_nopos_randomforest_R0.csv 
#python classification_benchmark.py -v ../vectors/ohsumed_global/norm_nopos_wrongnames_R0_9/R0 $algconf -r ../results/ohsumed_global_norm_nopos_randomforest_R0.csv 

#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-knn -m bm25 -r ../results/reuters21578_bench_randomforest_bm25.csv &
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-knn -m tfig -r ../results/reuters21578_bench_randomforest_tfig.csv &
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-knn -m tfchi2 -r ../results/reuters21578_bench_randomforest_tfchi2.csv & 
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-knn -m sublinear_tf -r ../results/reuters21578_bench_randomforest_sublinear_tf.csv & 
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-knn -m sublinear_tfidf -r ../results/reuters21578_bench_randomforest_sublinear_tfidf.csv & 
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-knn -m hashing -r ../results/reuters21578_bench_randomforest_sublinear_hashing.csv &

#python classification_benchmark.py -d 20newsgroups --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -r ../results/20newsgroups_bench_knn_grid.csv &
#python classification_benchmark.py -d ohsumed --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -r ../results/ohsumed_bench_knn_grid.csv &

#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -m tfgr -r ../results/reuters21578_bench_knn_grid_tfgr.csv 
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -m tfidf -r ../results/reuters21578_bench_knn_grid_tfidf.csv 
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -m count -r ../results/reuters21578_bench_knn_grid_count.csv 
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -m binary -r ../results/reuters21578_bench_knn_grid_binary.csv 
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -m sublinear_tfidf -r ../results/reuters21578_bench_knn_grid_sublinear_tfidf.csv 

#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -m sublinear_tf -r ../results/reuters21578_bench_knn_grid_sublinear_tf.csv 
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -m tfchi2 -r ../results/reuters21578_bench_knn_grid_tfchi2.csv 
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -m tfig -r ../results/reuters21578_bench_knn_grid_tfig.csv 
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -m tfrf -r ../results/reuters21578_bench_knn_grid_tfrf.csv 
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -m bm25 -r ../results/reuters21578_bench_knn_grid_bm25.csv 
#python classification_benchmark.py -d reuters21578 --no-linearsvm --no-multinomialnb --no-logisticregression --no-randomforest -m hashing -r ../results/reuters21578_bench_knn_grid_hashing.csv 

# relaunch RF and KNN for GR (running)
# relaunch RF and KNN for CHI (pending)
# relaunch RF and KNN for RelFreq
# launch all learners for ConfWeight

for dataset in 'ohsumed' 'reuters21578' '20newsgroups'
do
	python classification_benchmark.py -d $dataset --no-linearsvm --no-multinomialnb --no-logisticregression -m tfgr -r ../results/"$dataset"_bench_randforest_knn_gr.csv 
	python classification_benchmark.py -d $dataset --no-linearsvm --no-multinomialnb --no-logisticregression -m tfchi2 -r ../results/"$dataset"_bench_randforest_knn_chi.csv 
	#python classification_benchmark.py -d $dataset --no-linearsvm --no-multinomialnb --no-logisticregression -m tfcw -r ../results/"$dataset"_bench_randforest_knn_conf.csv 
	#python classification_benchmark.py -d $dataset --no-linearsvm --no-multinomialnb --no-logisticregression -m tfrf -r ../results/"$dataset"_bench_randforest_knn_rf.csv 
	wait
done



