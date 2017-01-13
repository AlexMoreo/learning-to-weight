import os, sys
import time
from time import gmtime, strftime
import pandas as pd
import numpy as np

class ReusltTable:
    def __init__(self, result_container):
        self.result_container = result_container

        if os.path.exists(result_container):
            self.df = pd.read_csv(result_container)
        else:
            self.df = pd.DataFrame(columns=['classifier',  # linearsvm, random forest, Multinomial NB,
                                       'vectorizer',  # binary, count, tf, tfidf, tfidf sublinear, bm25, hashing, learnt
                                       'num_features',
                                       'dataset',  # 20newsgroup, rcv1, ...
                                       'category',
                                       'run',
                                       'date',
                                       'time',
                                       'elapsedtime',
                                       'hiddensize',
                                       'lrate',
                                       'optimizer',
                                       'normalize',
                                       'nonnegative',
                                       'pretrain',
                                       'iterations',
                                       'notes',
                                       'acc', 'fscore', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn'])

    def add_empty_entry(self):
        self.df.loc[len(self.df)] = [np.nan] * len(self.df.columns)

    def set(self, column, value):
        if isinstance(value, float):
            value = float('%.3f'%value)
        self.df.iloc[len(self.df) - 1, list(self.df.columns).index(column)] = value

    def get(self, column):
        return self.df.iloc[len(self.df) - 1, list(self.df.columns).index(column)]

    def append(self, column, value):
        prev = self.get(column)
        self.set(column, str(prev) + " " + str(value))

    def set_all(self, dictionary):
        for key,value in dictionary.items():
            self.set(key,value)

    def commit(self):
        self.df.to_csv(self.result_container, index=False)

    def init_row_result(self, classifier_name, data, run=0):
        self.add_empty_entry()
        self.set('classifier', classifier_name)
        self.set('vectorizer', data.vectorize)
        self.set('num_features', data.num_features())
        self.set('dataset', data.name)
        self.set('category', data.positive_cat)
        self.set('run', run)

    def add_result_metric_scores(self, acc, f1, prec, rec, cont_table, init_time, notes=''):
        self.append('notes', notes)
        self.set_all({'acc': acc, 'fscore': f1, 'precision': prec, 'recall': rec})
        self.set_all(cont_table)
        self.set_all({'date': strftime("%d-%m-%Y", gmtime()), 'time': init_time, 'elapsedtime': time.time() - init_time})