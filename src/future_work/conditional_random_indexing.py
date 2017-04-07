import sys
import math
from sklearn import random_projection
from future_work.random_indexing import RandomIndexing
import random as rn
#def warn(*args, **kwargs):
#    pass
#import warnings
#warnings.warn = warn
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

def linear_svm_without_gridsearch(data):
    model = OneVsRestClassifier(LinearSVC())

    Xtr, ytr = data.get_devel_set()
    trained_model = model.fit(Xtr, ytr)

    Xte, yte = data.get_test_set()
    yte_ = trained_model.predict(Xte)

    macro_f1 = macroF1(yte, yte_)
    micro_f1 = microF1(yte, yte_)
    print("Test scores: %.3f macro-f1, %.3f micro-f1" % (macro_f1, micro_f1))


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


class ConditionalRandomIndexing(RandomIndexing):

    def __init__(self, latent_dimensions, non_zeros=2, positive=False, prob_interpretation="l1"):
        if prob_interpretation not in ["l1", "softmax"]:
            raise ValueError("Error: probabilistic interpretation should be either 'l1' or 'softmax'")
        self.prob_interpretation=prob_interpretation
        super(ConditionalRandomIndexing, self).__init__(latent_dimensions, non_zeros, positive)

    # takes a distribution of events (e.g., column of a coocurrence matrix indicating the amount of times the feature is
    # present in each document) and transforms it into a multinomial distribution; this transformation could be carried out
    # through an L1 normalization (default) or a softmax
    def interpret_distribution_as_multinomial(self, x):
        if self.prob_interpretation == "l1":
            norm = np.sum(x)
            if norm > 0:
                return x / norm
            return x
        elif self.prob_interpretation == "softmax":
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

    #the round_dim prevents empty dimensions
    def get_conditional_random_index(self, conditional_distribution):
        val = 1.0 / math.sqrt(self.non_zeros)
        sampled_dims = np.random.multinomial(self.non_zeros, conditional_distribution)
        random_vector = sampled_dims * val
        if not self.positive:
            length = conditional_distribution.shape[0]
            random_vector[np.random.permutation(length)[:length/2]] *= -1
        return random_vector

    def sample_coocurrence_events(self, coocurrence_matrix, n_samples):
        nD, nF = coocurrence_matrix.shape
        all_doc_indexes = range(nD)
        # repeats all document-index whenever possible, and fills the remainder with a permutation
        # eg nD=4, n_samples=11 -> [0,1,2,3]+[0,1,2,3]+[2,1,3]
        selection = all_doc_indexes * (n_samples/nD) + np.random.permutation(nD)[:n_samples%nD].tolist()
        return coocurrence_matrix[selection, :].toarray()

    def get_random_index_dictionary(self, coocurrence_matrix):
        nD,nF = coocurrence_matrix.shape
        sampled_coocurrence = self.sample_coocurrence_events(coocurrence_matrix, self.latent_dimensions)
        return {i:self.get_conditional_random_index(self.interpret_distribution_as_multinomial(sampled_coocurrence[:,i]))
                for i in range(nF)}


data = TextCollectionLoader(dataset='reuters21578')



nD = data.num_devel_documents()
nF = data.num_features()

nR = 5000
# nR = nF #without dimensionality reduction
non_zeros = 2

random_indexing = RandomIndexing(latent_dimensions=nR, non_zeros=non_zeros, positive=False)
#random_indexing = ConditionalRandomIndexing(latent_dimensions=nR, non_zeros=non_zeros, positive=False, prob_interpretation="softmax")

if False:
    print "Gaussian random projection"
    print random_indexing.count_nonzeros(data.devel_vec), random_indexing.density(data.devel_vec)
    #transformer = random_projection.SparseRandomProjection(n_components=nR, density=2.0/nR)
    transformer = random_projection.SparseRandomProjection(n_components=nR)
    data.devel_vec = transformer.fit_transform(data.devel_vec)
    data.test_vec = transformer.transform(data.test_vec)
    print data.devel_vec.shape
    print random_indexing.count_nonzeros(data.devel_vec), random_indexing.density(data.devel_vec)
    print "classify"
    linear_svm_without_gridsearch(data)
    sys.exit()
    #0.533 macro-f1, 0.833 micro-f1 with density density=2.0/nR
    #0.556 macro-f1, 0.849 micro-f1

print "random indexing"
print random_indexing.count_nonzeros(data.devel_vec), random_indexing.density(data.devel_vec)
data.devel_vec = random_indexing.fit_transform(data.devel_vec)
data.test_vec   = random_indexing.transform(data.test_vec)
print random_indexing.count_nonzeros(data.devel_vec), random_indexing.density(data.devel_vec)

#print "classification / test"
linear_svm_without_gridsearch(data)

#coocurrence:           Test scores: 0.583 macro-f1, 0.854 micro-f1
# LRI, nR=nF/2 (k=2)    Test scores: 0.590 macro-f1, 0.853 micro-f1
# LRI, nR=nF/2 (k=2)    Test scores: 0.594 macro-f1, 0.855 micro-f1
# RI,  nR=nF/2 (k=5)    Test scores: 0.584 macro-f1, 0.852 micro-f1
# CRI, nR=nF/2 (k=2) L1 Test scores: 0.578 macro-f1, 0.846 micro-f1

#nF, k=10 RI Test scores: 0.562 macro-f1, 0.852 micro-f1
#nF, k=2  RI Test scores: 0.564 macro-f1, 0.855 micro-f1
#5000, k=2  RI Test scores: 0.550 macro-f1, 0.843 micro-f1
#nF, k=10 CRI Test scores: 0.560 macro-f1, 0.850 micro-f1
#nF, k=2  CRI Test scores: 0.562 macro-f1, 0.854 micro-f1
#5000 k=2  CRI Test scores: 0.550 macro-f1, 0.846 micro-f1

