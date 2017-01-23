import numpy as np
import math, time
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_selection_function import *
import dill
from joblib import Parallel, delayed
import multiprocessing


class BM25:

    def __init__(self, k1=1.2, b=0.75, stop_words=None, min_df=1):
        self.k1 = k1
        self.b = b
        self.stop_words = stop_words
        self.min_df = min_df

    def fit(self, raw_documents):
        self.vectorizer = CountVectorizer(stop_words=self.stop_words, min_df=self.min_df)
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


def wrap_contingency_table(f, feat_vec, cat_doc_set, nD):
    feat_doc_set = set(feat_vec[:,f].nonzero()[0])
    return feature_label_contingency_table(cat_doc_set, feat_doc_set, nD)

class TftsrVectorizer:
    def __init__(self, binary_target, tsr_function, stop_words='english', sublinear_tf=False, min_df=1):
        self.stop_words = stop_words
        self.sublinear_tf = sublinear_tf
        self.tsr_function = tsr_function
        self.cat_doc_set = set([i for i, label in enumerate(binary_target) if label == 1])
        self.supervised_info = None
        self.min_df = min_df

    def fit_transform(self, raw_documents):
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words, sublinear_tf=self.sublinear_tf, use_idf=False, min_df=self.min_df)
        self.devel_vec = self.vectorizer.fit_transform(raw_documents)
        #return self.devel_vec
        #print self.devel_vec.shape
        return self.supervised_weighting(self.devel_vec)

    def transform(self, raw_documents):
        if not hasattr(self, 'vectorizer'): raise NameError('TftsrVectorizer: transform method called before fit.')
        transformed = self.vectorizer.transform(raw_documents)
        return self.supervised_weighting(transformed)
        #return transformed

    def supervised_weighting(self, w):
        w = w.toarray()
        #num_cores = multiprocessing.cpu_count()
        nD, nF = w.shape
        if self.supervised_info is None:
            sup = [wrap_contingency_table(f,self.devel_vec, self.cat_doc_set, nD) for f in range(w.shape[nF])]
            #sup = Parallel(n_jobs=num_cores, backend="threading")(
            #    delayed(wrap_contingency_table)(f, self.devel_vec, self.cat_doc_set, nD) for f in range(nF))
            self.supervised_info = np.array([self.tsr_function(sup_i) for sup_i in sup])
        sup_w = np.multiply(w, self.supervised_info)
        sup_w = sklearn.preprocessing.normalize(sup_w, norm='l2', axis=1, copy=False)
        return scipy.sparse.csr_matrix(sup_w)


