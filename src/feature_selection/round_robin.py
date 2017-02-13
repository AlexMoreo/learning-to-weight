import sys
import numpy as np
from joblib import Parallel, delayed
from tsr_function import *
import time

class RoundRobin:
    def __init__(self, k, score_func=information_gain, supervised_4cell_matrix=None):
        self._score_func = score_func
        self._k = k
        self.supervised_4cell_matrix = supervised_4cell_matrix

    def fit(self, X, y):
        print("Round Robin")
        nF = X.shape[1]
        nC = y.shape[1]
        print "Getting supervised 4 cell matrix"
        if self.supervised_4cell_matrix == None:
            self.supervised_4cell_matrix = get_supervised_matrix(X, y)
        print "Getting tsr matrix"
        tsr_matrix = get_tsr_matrix(self.supervised_4cell_matrix, self._score_func)

        #enhance the tsr_matrix with the feature index
        print "Enhancing tsr matrix"
        tsr_matrix = [[(tsr_matrix[c,f], f) for f in range(nF)] for c in range(nC)]

        print "Sorting tsr matrix"
        for c in range(nC):
            tsr_matrix[c].sort(key=lambda x: x[0]) #sort by tsr

        print "Selecting bests"
        sel_feats = set()
        round = 0
        while len(sel_feats) < self._k:
            sel_feats.add(tsr_matrix[round].pop()[1])
            round = (round+1) % nC

        self._best_feats = list(sel_feats)
        self._best_feats.sort()
        self.supervised_4cell_matrix = self.supervised_4cell_matrix[:,self._best_feats]

    def transform(self, X):
        if not hasattr(self, '_best_feats'): raise NameError('Transform method called before fit.')
        return X[:, self._best_feats]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)



