import multiprocessing

import numpy as np
import scipy
import sklearn
import math
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from feature_selection.tsr_function import *


class BM25:

    def __init__(self, k1=1.2, b=0.75, stop_words=None, min_df=1):
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(stop_words=stop_words, min_df=min_df)
        self.bm25transformer = BM25Transformer(k1=self.k1, b=self.b)

    def fit(self, raw_documents):
        self.train_matrix = self.vectorizer.fit_transform(raw_documents)

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.bm25transformer.fit_transform(self.train_matrix)

    def transform(self, raw_documents):
        tf = self.vectorizer.transform(raw_documents)
        return self.bm25transformer.transform(tf)

class BM25Transformer:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b

    def fit(self, coocurrence_matrix):
        self.tf = coocurrence_matrix.asfptype().toarray()
        self.nD = self.tf.shape[0]
        self.avgdl = []
        for d in range(self.nD): self.avgdl.append(self.tf[d, :].sum())
        self.avgdl = sum(self.avgdl) * 1.0 / len(self.avgdl)
        self.idf = dict()
        for f in range(self.tf.shape[1]):
            self.idf[f] = self._idf(self.nD, len(self.tf[:, f].nonzero()[0]))

    def fit_transform(self, coocurrence_matrix):
        self.fit(coocurrence_matrix)
        return self.transform_tf(self.tf)

    def transform(self, coocurrence_matrix):
        if not hasattr(self, 'idf'): raise NameError('BM25: transform method called before fit.')
        tf = coocurrence_matrix.asfptype().toarray()
        return self.transform_tf(tf)

    def transform_tf(self, tf):
        nD, nF = tf.shape
        for d in range(nD):
            len_d = tf[d, :].sum()
            norm = 0.0
            for f in tf[d].nonzero()[0]:
                tf[d, f] = self._score(tf[d, f], self.idf[f], self.k1, self.b, len_d, self.avgdl)
                norm += (tf[d, f] * tf[d, f])
            tf[d, :] /= math.sqrt(norm) if norm > 0 else 1
        return scipy.sparse.csr_matrix(tf)

    def _score(self, tfi, idfi, k1, b, len_d, avgdl):
        return idfi * (tfi * (k1 + 1) / (tfi + k1 * (1 - b + b * len_d / avgdl)))

    def _idf(self, nD, nd_fi):
        return max(math.log((nD - nd_fi + 0.5) / (nd_fi + 0.5)), 0.0)

def wrap_contingency_table(f, feat_vec, cat_doc_set, nD):
    feat_doc_set = set(feat_vec[:,f].nonzero()[0])
    return feature_label_contingency_table(cat_doc_set, feat_doc_set, nD)

"""
Supervised Term Weighting function based on any Term Selection Reduction (TSR) function (e.g., information gain,
chi-square, etc.) or, more generally, on any function that could be computed on the 4-cell contingency table for
each category-feature pair.
The supervised_4cell_matrix (a CxF matrix containing the 4-cell contingency tables
for each category-feature pair) can be pre-computed (e.g., during the feature selection phase) and passed as an
argument.
When C>1, i.e., in multiclass scenarios, a global_policy is used in order to determine a single feature-score which
informs about its relevance. Accepted policies include "max" (takes the max score across categories), "ave" and "wave"
(take the average, or weighted average, across all categories -- weights correspond to the class prevalence), and "sum"
(which sums all category scores).
"""
class TSRweighting:
    def __init__(self, tsr_function, global_policy='max', supervised_4cell_matrix=None, sublinear_tf=True, n_jobs=-1):
        if global_policy not in ['max', 'ave', 'wave', 'sum']: raise ValueError('Global policy should be in {"max", "ave", "wave", "sum"}')
        self.tsr_function = tsr_function
        self.global_policy = global_policy
        self.supervised_4cell_matrix = supervised_4cell_matrix
        self.n_jobs=n_jobs
        self.sublinear_tf=sublinear_tf

    def fit(self, X, y):
        print("TSR weighting")
        if self.sublinear_tf:
            self.unsupervised_vectorizer = TfidfTransformer(norm=None, use_idf=False, smooth_idf=False, sublinear_tf=True).fit(X)
        if self.supervised_4cell_matrix is None:
            print("Computing the 4-cell matrix")
            self.supervised_4cell_matrix = get_supervised_matrix(X, y, n_jobs=self.n_jobs)
        else:
            nC=y.shape[1]
            nF = X.shape[1]
            if self.supervised_4cell_matrix.shape != (nC, nF): raise ValueError("Shape of supervised information matrix is inconsistent with X and y")
        print("Computing the TSR matrix")
        tsr_matrix = get_tsr_matrix(self.supervised_4cell_matrix, self.tsr_function)
        if self.global_policy == 'ave':
            self.global_tsr_matrix = np.average(tsr_matrix, axis=0)
        elif self.global_policy == 'wave':
            nD,nC=y.shape
            category_prevalences = [sum(y[:,c])*1.0/nD for c in range(nC)]
            self.global_tsr_matrix = np.average(tsr_matrix, axis=0, weights=category_prevalences)
        elif self.global_policy == 'sum':
            self.global_tsr_matrix = np.sum(tsr_matrix, axis=0)
        elif self.global_policy == 'max':
            self.global_tsr_matrix = np.amax(tsr_matrix, axis=0)

    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)

    def transform(self, X):
        if not hasattr(self, 'global_tsr_matrix'): raise NameError('TSRweighting: transform method called before fit.')
        if self.sublinear_tf:
            X = self.unsupervised_vectorizer.transform(X)
        X = X.toarray()
        weighted_X = np.multiply(X, self.global_tsr_matrix)
        weighted_X = sklearn.preprocessing.normalize(weighted_X, norm='l2', axis=1, copy=False)
        return scipy.sparse.csr_matrix(weighted_X)


