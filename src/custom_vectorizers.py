import numpy as np
import math, time
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_selection_function import *

class BM25:

    def __init__(self, k1=1.2, b=0.75, stop_words=None):
        self.k1 = k1
        self.b = b
        self.stop_words = stop_words

    def fit(self, raw_documents):
        self.vectorizer = CountVectorizer(stop_words=self.stop_words)
        tf = self.vectorizer.fit_transform(raw_documents).asfptype().toarray()
        self.nD = tf.shape[0]
        self.avgdl = []
        for d in range(self.nD): self.avgdl.append(tf[d,:].sum())
        self.avgdl = sum(self.avgdl) * 1.0 / len(self.avgdl)
        self.idf = dict()
        for f in range(tf.shape[1]):
            self.idf[f] = self._idf(self.nD, len(tf[:, f].nonzero()[0]))
        return tf

    def fit_transform(self, raw_documents):
        tf = self.fit(raw_documents)
        return self.transform_tf(tf)

    def transform(self, raw_documents):
        if not hasattr(self, 'vectorizer'): raise NameError('BM25: transform method called before fit.')
        tf = self.vectorizer.transform(raw_documents).asfptype().toarray()
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


class TftsrVectorizer:
    def __init__(self, binary_target, tsr_function, stop_words='english', sublinear_tf=False):
        self.stop_words = stop_words
        self.sublinear_tf = sublinear_tf
        self.tsr_function = tsr_function
        self.cat_doc_set = set([i for i, label in enumerate(binary_target) if label == 1])

    def fit_transform(self, raw_documents):
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words, sublinear_tf=self.sublinear_tf, use_idf=False)
        self.devel_vec = self.vectorizer.fit_transform(raw_documents)
        return self.supervised_weighting(self.devel_vec)

    def transform(self, raw_documents):
        if not hasattr(self, 'vectorizer'): raise NameError('TftsrVectorizer: transform method called before fit.')
        transformed = self.vectorizer.transform(raw_documents)
        return self.supervised_weighting(transformed)

    def supervised_weighting(self, w):
        sup = [self.feature_label_contingency_table(f) for f in range(w.shape[1])]
        pc = sup[0].p_c()
        feat_corr_info = np.array([self.tsr_function(sup_i.tpr(), sup_i.fpr(), pc) for sup_i in sup])
        sup_w = w.multiply(feat_corr_info)
        sup_w = sklearn.preprocessing.normalize(sup_w, norm='l2', axis=1, copy=False)
        return scipy.sparse.csr_matrix(sup_w)

    def feature_label_contingency_table(self, feat_index):
        feat_vec = self.devel_vec[:, feat_index]  # TODO: cache also the feature-vectors
        feat_doc_set = set(feat_vec.nonzero()[0])
        nD = self.devel_vec.shape[0]
        return feature_label_contingency_table(self.cat_doc_set, feat_doc_set, nD)
