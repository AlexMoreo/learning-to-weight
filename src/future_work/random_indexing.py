import dill
import math
import random as rn
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

class RandomIndexing(object):

    def __init__(self, latent_dimensions, non_zeros=2, positive=False):
        self.latent_dimensions = latent_dimensions
        self.non_zeros = non_zeros
        self.positive = positive

    #the round_dim prevents empty dimensions
    def get_random_index(self, round_dim=-1):
        non_zero_dim_value = {}
        val = 1.0 / math.sqrt(self.non_zeros)
        while len(non_zero_dim_value) < self.non_zeros:
            if round_dim != -1:
                rand_dim = round_dim
                round_dim = -1
            else:
                rand_dim = rn.randint(0, self.latent_dimensions-1)
            if rand_dim not in non_zero_dim_value:
                if self.positive:
                    non_zero_dim_value[rand_dim] = +val
                else:
                    non_zero_dim_value[rand_dim] = +val if rn.random() < 0.5 else -val
        rand_vector = np.zeros(self.latent_dimensions)
        rand_vector[non_zero_dim_value.keys()] = non_zero_dim_value.values()
        return rand_vector

    def get_random_index_dictionary(self, coocurrence_matrix):
        original_dim = coocurrence_matrix.shape[1]
        return {i: self.get_random_index(round_dim=i % self.latent_dimensions) for i in range(original_dim)}

    def _as_projection_matrix(self, random_dic):
        return csc_matrix(np.vstack([rand_vec for dim, rand_vec in sorted(random_dic.items())]))

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def fit(self, X, y=None):
        if not hasattr(self, "projection_matrix"):
            self.projection_matrix = self._as_projection_matrix(self.get_random_index_dictionary(X))
        else:
            raise ValueError("Error: projection matrix is already calculated.")
        return self

    def transform(self, X, y=None):
        if not hasattr(self, "projection_matrix"):
            raise ValueError("Error: transform method called before fit.")
        projection = X * self.projection_matrix
        if self.density(projection) < 0.05:
            #csc_matrix
            return projection
        else:
            #ndarray
            return projection.toarray()

    def count_nonzeros(self, matrix):
        if isinstance(matrix, csc_matrix) or isinstance(matrix, csr_matrix):
            return matrix.nnz
        else:
            return np.count_nonzero(matrix)

    def proj_density(self):
        if not hasattr(self, "projection_matrix"):
            raise ValueError("Error: density method called before fit.")
        return self.density(self.projection_matrix)

    def density(self, matrix):
        return self.count_nonzeros(matrix)*1.0/ np.prod(matrix.shape)


import sys
import math
from sklearn import random_projection
import random as rn
#def warn(*args, **kwargs):
#    pass
#import warnings
#warnings.warn = warn
import numpy as np
from data.dataset_loader import TextCollectionLoader
from sklearn.feature_extraction.text import *
from sklearn.svm import SVC, LinearSVC
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.multiclass import OneVsRestClassifier
from utils.metrics import *
from data.dataset_loader import TextCollectionLoader

def fit_model_hyperparameters(data, parameters, model):
    single_class = data.num_categories() == 1
    if not single_class:
        parameters = {'estimator__' + key: parameters[key] for key in parameters.keys()}
        model = OneVsRestClassifier(model, n_jobs=-1)
    model_tunning = GridSearchCV(model, param_grid=parameters,
                                 scoring=make_scorer(macroF1), error_score=0, refit=True, cv=3, n_jobs=-1)

    Xtr, ytr = data.get_devel_set()
    Xtr.sort_indices()
    if single_class:
        ytr = np.squeeze(ytr)

    tunned = model_tunning.fit(Xtr, ytr)

    return tunned

def fit_and_test_model(data, parameters, model):
    init_time = time.time()
    tunned_model = fit_model_hyperparameters(data, parameters, model)
    tunning_time = time.time() - init_time
    print("%s: best parameters %s, best score %.3f, took %.3f seconds" %
          (type(model).__name__, tunned_model.best_params_, tunned_model.best_score_, tunning_time))

    Xte, yte = data.get_test_set()
    Xte.sort_indices()
    yte_ = tunned_model.predict(Xte)

    macro_f1 = macroF1(yte, yte_)
    micro_f1 = microF1(yte, yte_)
    print("Test scores: %.3f macro-f1, %.3f micro-f1" % (macro_f1, micro_f1))
#print "classification / test"
def linear_svm(data):
    parameters = {'C': [1e2, 1e1, 1],
                  'loss': ['squared_hinge'],
                  'dual': [True, False]}
    model = LinearSVC()
    fit_and_test_model(data, parameters, model)

def learner_without_gridsearch(data, learner):
    model = OneVsRestClassifier(learner, n_jobs=7)

    Xtr, ytr = data.get_devel_set()
    trained_model = model.fit(Xtr, ytr)

    Xte, yte = data.get_test_set()
    yte_ = trained_model.predict(Xte)

    macro_f1 = macroF1(yte, yte_)
    micro_f1 = microF1(yte, yte_)
    print("Test scores: %.3f macro-f1, %.3f micro-f1" % (macro_f1, micro_f1))

data = TextCollectionLoader(dataset='reuters21578')
data.devel.target = data.devel.target[:,:10]
data.test.target = data.test.target[:,:10]

print("\nwith bow and linear kernel")
learner_without_gridsearch(data, LinearSVC())

nD = data.num_devel_documents()
nF = data.num_features()
nR = nF
non_zeros = 2 # nF*0.01
random_indexing = RandomIndexing(latent_dimensions=nR, non_zeros=non_zeros, positive=False)

data._devel_vec  = random_indexing.fit_transform(data._devel_vec)
data._test_vec   = random_indexing.transform(data._test_vec)

P = random_indexing.projection_matrix
R = np.dot(P.T,P)
ri_kernel = lambda X, Y: np.dot(np.dot(X, R), Y.T).toarray()

print("\nwith linear kernel")
learner_without_gridsearch(data, LinearSVC())

print("\nwith self-covariance kernel")
learner_without_gridsearch(data, SVC(kernel=ri_kernel))

#


#BoW   Test scores: 0.564 macro-f1, 0.855 micro-f1
#RI 1% Test scores: 0.559 macro-f1, 0.848 micro-f1
#LRI   Test scores: 0.542 macro-f1, 0.842 micro-f1
#LRI K:Test scores: 0.565 macro-f1, 0.842 micro-f1

# with bow and linear kernel
# Test scores: 0.564 macro-f1, 0.855 micro-f1
#
# with linear kernel
# Test scores: 0.561 macro-f1, 0.855 micro-f1
#
# with self-covarianze kernel
# Test scores: 0.578 macro-f1, 0.859 micro-f1