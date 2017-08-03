from __future__ import print_function
import utils.disable_sklearn_warnings
from pprint import pprint
from data.custom_vectorizers import TfidfTransformerAlphaBeta, BM25TransformerAlphaBeta, TSRweightingAlphaBeta
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from data.dataset_loader import TextCollectionLoader
from feature_selection.tsr_function import *
from utils.metrics import *
import sys,os
import pandas as pd
import argparse
import numpy as np
from utils.helpers import err_exception
from utils.result_table import AB_Results

# ver la distribucion de tf y la distribuicion de idf
# fisher score and PMI and GSS and IG(pos)

class AB_Results_Local(AB_Results):
    def __init__(self, file, autoflush=True, verbose=False):
        columns = ['learner', 'dataset', 'vectorizer', 'nF', 'cat', 'bestparams', 'alpha', 'beta', 'f1', 'tp', 'tn', 'fp', 'fn']
        super(AB_Results_Local, self).__init__(file=file, columns=columns, autoflush=autoflush, verbose=verbose)

def get_learner_and_params():
    if args.learner == 'LinearSVC':
        clf = LinearSVC()
        clf_params = {'clf__C': [10 ** i for i in range(0, 3)]}
    else:
        print("Learner {} is not supported. Exit.".format(args.learner))

    return clf, clf_params

def get_vectorizer():
    tsr_map = {'tfig': information_gain, 'tfchi': chi_square, 'tfgr': gain_ratio, 'tfrf': relevance_frequency,
               'tfcw': conf_weight, 'tfgss':gss, 'tfpmi':pointwise_mutual_information, 'tfigpos':positive_information_gain}
    if args.vectorizer == 'tfidf':
        vect = TfidfTransformerAlphaBeta(sublinear_tf=args.sublinear_tf, use_idf=True, norm='l2')
    elif args.vectorizer == 'bm25':
        err_exception(args.sublinear_tf, 'Logarithmic version of BM25 is not available. Exit.')
        vect = BM25TransformerAlphaBeta(norm='none')
    elif args.vectorizer in weight_functions:
        supervised_vector = supervised_matrix[cat:cat + 1, :]
        vect = TSRweightingAlphaBeta(tsr_function=tsr_map[args.vectorizer], sublinear_tf=args.sublinear_tf,
                                     supervised_4cell_matrix=supervised_vector)
    else:
        print("Vectorizer {} is not supported. Exit.".format(args.vectorizer))

    return vect

def train_and_predict(Xtr, ytr_c, Xte):
    vect = get_vectorizer()
    Xtr_ = vect.fit_transform(Xtr, ytr_c)
    Xte_ = vect.transform(Xte)
    clf, _ = get_learner_and_params()
    clf.fit(Xtr_, ytr_c)
    return clf.predict(Xte_)

if __name__ == "__main__":

    unsupervised_weight_functions = ['tfidf', 'bm25']
    supervised_weight_functions = ['tfig', 'tfgr', 'tfchi', 'tfrf', 'tfcw', 'tfgss', 'tfpmi', 'tfigpos']
    weight_functions = unsupervised_weight_functions + supervised_weight_functions
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="indicates the dataset on which to run the baselines benchmark ", choices=TextCollectionLoader.valid_datasets)
    parser.add_argument("-r", "--resultfile", help="path to a result container file (.csv)", type=str, default="../results.csv")
    parser.add_argument("-v", "--vectorizer", help="selects one single vectorizer method to run from", type=str, choices=weight_functions)
    parser.add_argument("--fs", help="feature selection ratio", type=float, default=0.1)
    parser.add_argument("--sublinear_tf", help="logarithmic version of the tf-like function", default=False, action="store_true")
    parser.add_argument("-l", "--learner", help="learner", type=str, default='LinearSVC')
    parser.add_argument("-p", "--params", help="select the parameters to explore (0, 1, 2)", type=int, default=2)
    parser.add_argument("--recompute", help="even if the result was already computed, it is recomputed", default=False, action="store_true")
    args = parser.parse_args()

    err_exception(args.params < 0 or args.params > 2, "--params should be in [0,1,2]")
    print('Exploring {} parameters'.format(args.params))

    results = AB_Results_Local(args.resultfile, autoflush=True, verbose=True)
    data = TextCollectionLoader(dataset=args.dataset, vectorizer='tf', rep_mode='sparse', feat_sel=args.fs, sublinear_tf=False)
    nF = data.num_features()
    nC = data.num_categories()

    vectorizer_name = ('Log' if args.sublinear_tf else '') + args.vectorizer
    if args.params == 2:
        vectorizer_name = 'AB' + vectorizer_name
    elif args.params == 1:
        vectorizer_name = 'B' + vectorizer_name

    Xtr, ytr = data.get_devel_set()
    Xte, yte = data.get_test_set()

    if args.vectorizer in supervised_weight_functions:
        supervised_matrix = get_supervised_matrix(Xtr, ytr)

    for cat in range(nC):
        if not args.recompute and results.already_calculated(learner=args.learner, dataset=args.dataset, vectorizer=vectorizer_name, nF=nF, cat=cat):
            print("Experiment ({} {} {} {}) has been already calculated. Skipping it.".format(args.learner, args.dataset, vectorizer_name, nF))
            continue

        print("{}: cat={}".format(args.dataset, cat))

        ytr_c = ytr[:, cat]
        yte_c = yte[:, cat]
        prev_c = sum(ytr_c)
        print(prev_c)
        if prev_c == 1:
            # there seems to be a bug in scikit-learn regarding the gridsearch with only one positive example; fairly enough, it
            # cannot create the folds to optimize the parameters (though it should not exploit either...), so whenever this happens
            # we will simply train and predict, with default parameters
            yte_c_ = train_and_predict(Xtr, ytr_c, Xte)
            _4cell = single_metric_statistics(yte_c, yte_c_)
            fscore = f1(_4cell)
            results.add_row(learner = args.learner, dataset = args.dataset, vectorizer = vectorizer_name, nF = nF, cat = cat,
                            bestparams='indetermined', alpha =1.0, beta =1.0, f1 =fscore, tp = _4cell.tp, tn = _4cell.tn, fp = _4cell.fp, fn = _4cell.fn)
            continue

        vect = get_vectorizer()

        clf, clf_params = get_learner_and_params()

        pipeline = Pipeline([
            ('tfidf', vect),
            ('clf', clf),
        ])

        parameters = {
            'tfidf__alpha': [1.0] if args.params < 2 else np.linspace(0.25, 2.0, 8),
            'tfidf__beta': [1.0] if args.params < 1 else np.linspace(0.25, 2.0, 8),
        }
        parameters.update(clf_params)

        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=min(5,prev_c))#, scoring=make_scorer(macroF1))
        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint(parameters)

        t0 = time.time()
        grid_search.fit(Xtr, ytr_c)
        print("done in %0.3fs" % (time.time() - t0))
        print()

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        print("Running test and evaluation:")
        y_ = grid_search.predict(Xte)
        _4cell = single_metric_statistics(yte_c, y_)
        fscore = f1(_4cell)
        print("F1={}".format(fscore))

        best_alpha = best_parameters['tfidf__alpha']
        best_beta  = best_parameters['tfidf__beta']
        rest_parameters = ', '.join([k+'='+str(best_parameters[k]) for k in parameters.keys() if k not in ['tfidf__alpha', 'tfidf__beta']])

        results.add_row(learner=args.learner, dataset=args.dataset, vectorizer=vectorizer_name, nF=nF, cat=cat,
                        bestparams=str(rest_parameters), alpha=best_alpha, beta=best_beta, f1=fscore,
                        tp=_4cell.tp, tn=_4cell.tn, fp=_4cell.fp, fn=_4cell.fn)
