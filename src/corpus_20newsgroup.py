from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from helpers import *
import numpy as np
import time

class Dataset:
    def __init__(self, valid_proportion=0.1, categories=None, vectorize='hashing', delete_metadata=True, dense=False, positive_cat=None):
        err_exit(vectorize not in ['hashing', 'tfidf', 'count', 'binary'], err_msg='Param vectorize should be one in "hashing", "tfidf", "count", or "binary"')
        self.vectorize=vectorize
        self.dense=dense
        metadata = ('headers', 'footers', 'quotes') if delete_metadata else None
        self.devel = fetch_20newsgroups(subset='train', categories=categories, remove=metadata)
        self.test  = fetch_20newsgroups(subset='test',  categories=categories, remove=metadata)
        self.epoch = 0
        self.offset = 0
        self.positive_cat = positive_cat
        self.devel_vec, self.test_vec = self._vectorize_documents(vectorize)
        self.devel_indexes = self._get_doc_indexes(self.devel_vec)
        tr_val_split_point = int(self.num_devel_docs() * (1.0-valid_proportion))
        self.train_indexes = self.devel_indexes[:tr_val_split_point]
        self.valid_indexes = self.devel_indexes[tr_val_split_point:]
        self.test_indexes  = self._get_doc_indexes(self.test_vec)
        if self.dense:
            self.devel_vec = self.devel_vec.todense()
            self.test_vec = self.test_vec.todense()
            self._batch_getter = self._dense_batch_getter
        else:
            #already sparse representation
            self._batch_getter = self._sparse_batch_getter
        if self.positive_cat:
            err_exit(self.positive_cat not in self.devel.target_names, 'Error. Positive category not in scope.')
            self.binarize_classes()
        self.cat_vec_dic = dict()

    # change class codes: positive class = 1, all others = 0, and set category names to 'positive' or 'negative'
    def binarize_classes(self):
        pos_cat_code = self.devel.target_names.index(self.positive_cat)
        self.devel.target_names = self.test.target_names = ['negative', 'positive']
        def __binarize_codes(target, pos_code):
            target[target == pos_code] = -1
            target[target != -1] = 0
            target[target == -1] = 1
        __binarize_codes(self.devel.target, pos_cat_code)
        __binarize_codes(self.test.target,  pos_cat_code)

    def class_prevalence(self, cat_label=1):
        return sum(1.0 for x in self.devel.target if x == cat_label) / self.num_tr_documents()

    def _vectorize_documents(self, vectorize):
        if vectorize == 'hashing':
            vectorizer = HashingVectorizer(n_features=10000, stop_words='english')
            self.weight_getter = self._get_none
        elif vectorize == 'tfidf':
            vectorizer = TfidfVectorizer(stop_words='english')
            self.weight_getter = self._get_weights
        elif vectorize == 'count':
            vectorizer = CountVectorizer(stop_words='english')
            self.weight_getter = self._get_weights
        else: #binary
            vectorizer = CountVectorizer(stop_words='english', binary=True)
            self.weight_getter = self._get_none
        devel_vec = vectorizer.fit_transform(self.devel.data)
        test_vec = vectorizer.transform(self.test.data)
        #sorting the indexes simplifies the creation of sparse tensors a lot
        devel_vec.sort_indices()
        test_vec.sort_indices()
        return devel_vec, test_vec

    def is_weighted(self):
        return self.vectorize in ['tfidf', 'count']

    #virtually remove invalid documents (documents without any non-zero feature)
    def _get_doc_indexes(self, vector_set):
        all_indexes = np.arange(vector_set.shape[0])
        return [i for i in all_indexes if vector_set[i].nnz>0]

    def get_index_values(self, batch):
        num_indices = len(batch.nonzero()[0])
        indices = batch.nonzero()[0].reshape(num_indices, 1)
        values = batch.nonzero()[1]
        return indices, values

    def _get_none(self, batch):
        return []

    def _get_weights(self, batch):
        return batch.data

    def _sparse_batch_getter(self, batch):
        indices, values = self.get_index_values(batch)
        weights = self.weight_getter(batch)
        return indices, values, weights

    def _dense_batch_getter(self, batch):
        return batch


    def train_batch(self, batch_size=64):
        to_pos = min(self.offset + batch_size, self.num_tr_documents())
        batch = self.devel_vec[self.train_indexes[self.offset:to_pos]]
        batch_rep = self._batch_getter(batch)
        labels = self.devel.target[self.train_indexes[self.offset:to_pos]]
        self.offset += batch_size
        if self.offset >= self.num_tr_documents():
            self.offset = 0
            self.epoch += 1
            random.shuffle(self.train_indexes)
        return batch_rep, labels

    def val_batch(self):
        batch = self.devel_vec[self.valid_indexes]
        batch_rep = self._batch_getter(batch)
        labels = self.devel.target[self.valid_indexes]
        return batch_rep, labels

    def test_batch(self):
        batch = self.test_vec[self.test_indexes]
        batch_rep = self._batch_getter(batch)
        labels=self.test.target[self.test_indexes]
        return batch_rep, labels

    def feat_sup_statistics(self, feat_index, cat_label=1):
        feat_vec = self.devel_vec[:,feat_index]
        if cat_label not in self.cat_vec_dic:
            self.cat_vec_dic[cat_label] = [(1 if x==cat_label else 0) for x in self.devel.target]
        cat_vec  = self.cat_vec_dic[cat_label]
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(feat_vec)):
            if feat_vec[i]>0:
                if cat_vec[i] > 0: tp += 1
                else: fp += 1
            else:
                if cat_vec[i] > 0: fn += 1
                else: tn += 1
        return ContTable(tp=tp, tn=tn, fp=fp, fn=fn)

    def num_devel_docs(self):
        return len(self.devel_indexes)

    def num_categories(self):
        return len(np.unique(self.devel.target))

    def get_ategories(self):
        return np.unique(self.devel.target_names)

    def num_features(self):
        return self.devel_vec.shape[1]

    def num_tr_documents(self):
        return len(self.train_indexes)

    def num_val_documents(self):
        return len(self.valid_indexes)

    def num_test_documents(self):
        return len(self.test_indexes)

class ContTable:

    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        self.tp=tp
        self.tn=tn
        self.fp=fp
        self.fn=fn

    def get_d(self): return self.tp + self.tn + self.fp + self.fn

    def get_c(self): return self.tp + self.fn

    def get_not_c(self): return self.tn + self.fp

    def get_f(self): return self.tp + self.fp

    def get_not_f(self): return self.tn + self.fn

    def p_c(self): return (1.0*self.get_c())/self.get_d()

    def p_not_c(self): return 1.0-self.p_c()

    def p_f(self): return (1.0*self.get_f())/self.get_d()

    def p_not_f(self): return 1.0-self.p_f()

    def p_tp(self): return (1.0*self.tp)/self.get_d()

    def p_tn(self): return (1.0*self.tn) / self.get_d()

    def p_fp(self): return (1.0*self.fp) / self.get_d()

    def p_fn(self): return (1.0*self.fn) / self.get_d()

    def tpr(self):
        c = 1.0*self.get_c()
        return self.tp / c if c > 0.0 else 0.0

    def fpr(self):
        _c = 1.0*self.get_not_c()
        return self.fp / _c if _c > 0.0 else 0.0