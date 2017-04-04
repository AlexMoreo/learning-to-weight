import sys
import math
import random as rn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
from data.dataset_loader import TextCollectionLoader
from sklearn.feature_extraction.text import *
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.multiclass import OneVsRestClassifier
from utils.metrics import *


# -d reuters21578 -r ../results/baselines_svm.csv -m binary --fs 0.1 --classification binary

from data.dataset_loader import *
from data.weighted_vectors import WeightedVectors

def linear_svm(data):
    parameters = {'C': [1e2, 1e1, 1],
                  'loss': ['squared_hinge'],
                  'dual': [True, False]}
    model = LinearSVC()
    fit_and_test_model(data, parameters, model)

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

#the round_dim prevents empty dimensions
def get_random_index(dimensions, non_zeros=2, round_dim=-1):
    non_zero_dim_value = {}
    val = 1.0 / math.sqrt(non_zeros)
    while len(non_zero_dim_value) < non_zeros:
        if round_dim != -1:
            rand_dim = round_dim
            round_dim = -1
        else:
            rand_dim = rn.randint(0,dimensions-1)
        if rand_dim not in non_zero_dim_value:
            non_zero_dim_value[rand_dim] = +val if rn.random() < 0.5 else -val
    rand_vector = np.zeros(dimensions)
    rand_vector[non_zero_dim_value.keys()] = non_zero_dim_value.values()
    return rand_vector

def get_random_index_dictionary(original_dim, reduced_dim, non_zeros):
    return {i:get_random_index(dimensions=reduced_dim, non_zeros=non_zeros, round_dim=i%reduced_dim) for i in range(original_dim)}

def as_projection_matrix(random_dic):
    return np.vstack([rand_vec for dim, rand_vec in sorted(random_dic.items())])

def transform(cooc_matrix, projection_matrix):
    return csr_matrix(cooc_matrix * projection_matrix)


def colisions(projection_matrix):
    nF,nR = projection_matrix.shape
    return np.mean(np.sum(projection_matrix, axis=0))

data = TextCollectionLoader(dataset='reuters21578')

nD = data.num_devel_documents()
nF = data.num_features()

k = 2 #(int)(0.01 * nF)

#nR = nF #without dimensionality reduction
nR = nF / 2

print "getting dictionary"
random_dictionary = get_random_index_dictionary(nF, nR, k)
proj_matrix = as_projection_matrix(random_dictionary)
print "collisions", colisions(proj_matrix)

print "projecting"
data.devel_vec = transform(data.devel_vec, proj_matrix)
data.test_vec = transform(data.test_vec, proj_matrix)
print "done"

linear_svm(data)

#coocurrence: Test scores: 0.583 macro-f1, 0.854 micro-f1
# LRI, nR=nF/2Test scores: 0.585 macro-f1, 0.854 micro-f1




