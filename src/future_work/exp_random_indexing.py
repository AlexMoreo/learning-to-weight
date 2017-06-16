import dill
import sys
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
from sklearn.preprocessing import normalize
from utils.metrics import *
from data.dataset_loader import TextCollectionLoader
from future_work.random_indexing import RandomIndexing
import time

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
    #Xte.sort_indices()
    yte_ = tunned_model.predict(Xte)

    macro_f1 = macroF1(yte, yte_)
    micro_f1 = microF1(yte, yte_)
    return macro_f1, micro_f1

#print "classification / test"
def linear_svm(data, learner):
    parameters = {'C': [1e2, 1e1, 1],
                  'loss': ['squared_hinge'],
                  'dual': [True, False]}


def learner_without_gridsearch(Xtr,ytr,Xte,yte, learner, cat=-1):

    if cat!=-1:
        model = learner
        ytr = ytr[:,cat]
        yte = yte[:, cat]
    else:
        model = OneVsRestClassifier(learner, n_jobs=7)

    t_ini = time.time()
    trained_model = model.fit(Xtr, ytr)
    t_train = time.time()
    yte_ = trained_model.predict(Xte)
    t_test = time.time()
    return macroF1(yte, yte_), microF1(yte, yte_), t_train-t_ini, t_test-t_train

def sort_categories_by_prevalence(data):
    ytr, yte = data.devel.target, data.test.target
    nC = ytr.shape[1]
    ytr_prev = [np.sum(ytr[:,i], axis=0) for i in range(nC)]
    ordered = np.argsort(ytr_prev)
    data.devel.target = data.devel.target[:,ordered]
    data.test.target = data.test.target[:, ordered]



#-------------------------------------------------------------------------------------------------------------------

def svc_ri_kernel(random_indexer):
    P = random_indexer.projection_matrix
    R = np.absolute(P.transpose().dot(P))
    R.eliminate_zeros()
    if R.nnz*1.0/(R.shape[0]*R.shape[1]) > 0.1:
        R = R.todense()
        print('todense')
    #normalize(R, axis=1, norm='max', copy=False)
    ri_kernel = lambda X, Y: (X.dot(R)).dot(Y.T)
    return SVC(kernel=ri_kernel)

def experiment(Xtr,ytr,Xte,yte,learner,method,dataset,nF,fo,run=0):
    macro_f1, micro_f1, t_train, t_test = learner_without_gridsearch(Xtr,ytr,Xte,yte, learner)
    fo.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(method, dataset, nF, run, t_train, t_test, macro_f1, micro_f1))
    fo.flush()

with open('Kernel_RI_results.txt', 'w') as fo:
    fo.write("Method\tdataset\tnF\trun\ttrainTime\ttestTime\tMacroF1\tmicroF1\n")

    bow_computed = False
    for non_zeros in [2, 5, 10]:
        for dataset in TextCollectionLoader.valid_datasets:
            data = TextCollectionLoader(dataset=dataset)
            nF = data.num_features()

            if not bow_computed:
                X_train, y_train = data.get_devel_set()
                X_test,  y_test = data.get_test_set()
                experiment(X_train, y_train, X_test,  y_test, LinearSVC(), 'BoW', dataset, nF, fo)

            for run in range(5):
                for nR in [5000, 10000, nF/2, nF, int(nF*1.25)]:
                    print('Running {} nR={} non-zero={}...'.format(dataset,nR,non_zeros))
                    data = TextCollectionLoader(dataset=dataset)
                    X_train, y_train = data.get_devel_set()
                    X_test, y_test = data.get_test_set()

                    random_indexing = RandomIndexing(latent_dimensions=nR, non_zeros=non_zeros, positive=False, postnorm=True)
                    X_train = random_indexing.fit_transform(X_train)
                    X_test  = random_indexing.transform(X_test)

                    experiment(X_train, y_train, X_test, y_test, LinearSVC(), 'RI(%d)'%non_zeros, dataset, nR, fo, run=run)
                    experiment(X_train, y_train, X_test, y_test, svc_ri_kernel(random_indexing), 'KernelRI(%d)' % non_zeros, dataset, nR, fo, run=run)
        bow_computed = True
        print('Done!')