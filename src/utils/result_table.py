import os, sys
import time
from time import gmtime, strftime
import pandas as pd
import numpy as np

class BasicResultTable(object):
    def __init__(self, result_container, columns):
        self.result_container = result_container

        if os.path.exists(result_container):
            self.df = pd.read_csv(result_container)
        else:
            self.df = pd.DataFrame(columns=columns)

    def add_empty_entry(self):
        self.df.loc[len(self.df)] = [np.nan] * len(self.df.columns)

    def set(self, column, value):
        if isinstance(value, float):
            value = float('%.3f'%value)
        try:
            self.df.iloc[len(self.df) - 1, list(self.df.columns).index(column)] = value
        except ValueError:
            print("no index for attribute %s (pass)" %column)

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


class BaselineResultTable(BasicResultTable):
    columns = ['classifier',  # linearsvm, random forest, Multinomial NB,
                'weighting',  # binary, count, tf, tfidf, tfidf sublinear, bm25, hashing, learnt
                'num_features',
                'dataset',  # 20newsgroup, rcv1, ...
                'category',  # number in valid range, or "all"
                'date',
                'time',
                'elapsedtime',
                'notes']

    def __init__(self, result_container, classification_mode):
        if classification_mode not in ['binary', 'multiclass']:
            raise ValueError("Classification mode should be either binary or multiclass")
        columns = self.columns + self._evaluation_metrics(classification_mode)
        if result_container[-4:]=='.csv': result_container=result_container.replace('.csv', '.'+classification_mode+'.csv')
        else: result_container += classification_mode+'.csv'
        super(BaselineResultTable, self).__init__(result_container, columns)

    def _evaluation_metrics(self, classification_mode):
        if classification_mode == 'binary':
            return ['acc', 'fscore', 'tp', 'fp', 'fn', 'tn']
        else:
            return ['macro_f1', 'micro_f1']

    def init_row_result(self, classifier_name, data):
        self.add_empty_entry()
        self.set('classifier', classifier_name)
        self.set('weighting', data.vectorizer_name)
        self.set('num_features', data.num_features())
        self.set('dataset', data.name)
        self.set('category', data.positive_cat if data.positive_cat != None else "all")

    def add_result_scores_binary(self, acc, f1, cont_table, init_time, notes=''):
        self.append('notes', notes)
        self.set_all({'acc': acc, 'fscore': f1})
        self.set_all({'tp':cont_table.tp, 'fp':cont_table.fp, 'fn':cont_table.fn, 'tn':cont_table.tn})
        self.set_all({'date': strftime("%d-%m-%Y", gmtime()), 'time': init_time, 'elapsedtime': time.time() - init_time})

    def add_result_scores_multiclass(self, macro_f1, micro_f1, init_time, notes=''):
        self.append('notes', notes)
        self.set_all({'macro_f1': macro_f1, 'micro_f1': micro_f1})
        self.set_all({'date': strftime("%d-%m-%Y", gmtime()), 'time': init_time, 'elapsedtime': time.time() - init_time})

    def check_if_calculated(self, classifier, weighting, num_features, dataset, category):
        query_ = "classifier=='%s' and weighting=='%s' and num_features==%d and dataset=='%s' and category==%d" % (
            classifier, weighting, num_features, dataset, category
        )
        return len(self.df.query(query_)) > 0

class Learning2Weight_ResultTable(BaselineResultTable):
    learn_columns = ['run',
                    'hiddensize',
                    'learn_tf',
                    'learn_idf',
                    'learn_norm',
                    'iterations']
    def __init__(self, result_container, classification_mode):
        columns = self.columns + self._evaluation_metrics(classification_mode) + self.learn_columns
        super(BaselineResultTable, self).__init__(result_container, classification_mode, columns)

    def set_learn_params(self, hiddensize, iterations, learn_tf=False, learn_idf=False, learn_norm=False, run=0):
        self.set('hiddensize', hiddensize)
        self.set('iterations', iterations)
        self.set('learn_tf', learn_tf)
        self.set('learn_idf', learn_idf)
        self.set('learn_norm', learn_norm)
        self.set('run', run)