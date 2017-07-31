from __future__ import print_function
def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from pprint import pprint
from time import time
from data.custom_vectorizers import TfidfTransformerAlphaBeta, BM25TransformerAlphaBeta, TSRweightingAlphaBeta
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from data.dataset_loader import TextCollectionLoader
from sklearn.multiclass import OneVsRestClassifier
from feature_selection.tsr_function import information_gain, chi_square, relevance_frequency, conf_weight, gain_ratio, get_supervised_matrix
from utils.metrics import *
import sys,os
import pandas as pd
import argparse
import numpy as np

class AB_Results:
    def __init__(self, file, autoflush=True, verbose=False):
        self.file = file
        self.columns = ['learner', 'dataset', 'vectorizer', 'nF', 'bestparams', 'alpha', 'beta', 'MacroF1', 'microF1']
        self.autoflush = autoflush
        self.verbose = verbose
        if os.path.exists(file):
            self.tell('Loading existing file from {}'.format(file))
            self.df = pd.read_csv(file, sep='\t')
        else:
            self.tell('File {} does not exist. Creating new frame.'.format(file))
            dir = os.path.dirname(self.file)
            if dir and not os.path.exists(dir): os.makedirs(dir)
            self.df = pd.DataFrame(columns=self.columns)

    def already_calculated(self, learner, dataset, vectorizer, nF):
        return ((self.df['learner'] == learner) &
                (self.df['dataset'] == dataset) &
                (self.df['vectorizer'] == vectorizer) &
                (self.df['nF'] == nF)).any()

    def add_row(self, learner, dataset, vectorizer, nF, bestparams, alpha, beta, MacroF1, microF1):
        s = pd.Series([learner, dataset, vectorizer, nF, bestparams, alpha, beta, MacroF1, microF1], index=self.columns)
        self.df = self.df.append(s, ignore_index=True)
        if self.autoflush: self.flush()
        self.tell(s.to_string())

    def flush(self):
        self.df.to_csv(self.file, index=False, sep='\t')

    def tell(self, msg):
        if self.verbose: print(msg)

if __name__ == "__main__":

    weight_functions = ['tfidf', 'bm25', 'tfig', 'tfgr', 'tfchi', 'tfrf', 'tfcw']
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="indicates the dataset on which to run the baselines benchmark ", choices=TextCollectionLoader.valid_datasets)
    parser.add_argument("-r", "--resultfile", help="path to a result container file (.csv)", type=str, default="../results.csv")
    parser.add_argument("-v", "--vectorizer", help="selects one single vectorizer method to run from", type=str, choices=weight_functions)
    parser.add_argument("--fs", help="feature selection ratio", type=float, default=0.1)
    parser.add_argument("--sublinear_tf", help="logarithmic version of the tf-like function", default=False, action="store_true")
    parser.add_argument("--global_policy", help="global policy for supervised term weighting approaches", choices=['max', 'ave', 'wave', 'sum'], type=str, default='max')
    parser.add_argument("-l", "--learner", help="learner", type=str, default='LinearSVC')
    parser.add_argument("--not_explore_ab", help="deactivates the exploration of alpha and beta (eq. alpha=1, beta=1)", default=False, action="store_true")
    parser.add_argument("--recompute", help="even if the result was already computed, it is recomputed", default=False, action="store_true")
    args = parser.parse_args()

    results = AB_Results(args.resultfile, autoflush=True, verbose=True)
    data = TextCollectionLoader(dataset=args.dataset, vectorizer='tf', rep_mode='sparse', feat_sel=args.fs, sublinear_tf=False)
    nF = data.num_features()

    vectorizer_name = ('' if args.not_explore_ab else 'AB') + ('Log' if args.sublinear_tf else '') + args.vectorizer
    if not args.recompute and results.already_calculated(args.learner, args.dataset, vectorizer_name, nF):
        print("Experiment ({} {} {} {}) has been already calculated. Skipping it.".format(args.learner, args.dataset, vectorizer_name, nF))
        sys.exit(0)

    Xtr, ytr = data.get_devel_set()
    Xte, yte = data.get_test_set()

    tsr_map = {'tfig': information_gain, 'tfchi': chi_square, 'tfgr': gain_ratio, 'tfrf': relevance_frequency, 'tfcw': conf_weight}
    if args.vectorizer == 'tfidf':
        vect = TfidfTransformerAlphaBeta(sublinear_tf=args.sublinear_tf, use_idf=True, norm='l2')
    elif args.vectorizer == 'bm25':
        vect = BM25TransformerAlphaBeta(norm='none')
    elif args.vectorizer in weight_functions:
        supervised_matrix = get_supervised_matrix(Xtr, ytr)
        vect = TSRweightingAlphaBeta(tsr_function=tsr_map[args.vectorizer], global_policy=args.global_policy, sublinear_tf=args.sublinear_tf, supervised_4cell_matrix=supervised_matrix)
    else:
        print("Vectorizer {} is not supported. Exit.".format(args.vectorizer))

    if args.learner == 'LinearSVC':
        clf = OneVsRestClassifier(LinearSVC())
        clf_params = {'clf__estimator__C': [10 ** i for i in range(0, 4)]}
    else:
        print("Learner {} is not supported. Exit.".format(args.learner))

    pipeline = Pipeline([
        ('tfidf', vect),
        ('clf', clf),
    ])

    parameters = {
        'tfidf__alpha': [1.0] if args.not_explore_ab else np.linspace(0.25, 2.0, 8),
        'tfidf__beta': [1.0] if args.not_explore_ab else np.linspace(0.25, 2.0, 8),
    }
    parameters.update(clf_params)

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)

    t0 = time()
    grid_search.fit(Xtr, ytr)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print("Running test and evaluation:")
    y_ = grid_search.predict(Xte)
    macro_f1 = macroF1(yte, y_)
    micro_f1 = microF1(yte, y_)
    print("MacroF1={}, microF1={}".format(macro_f1, micro_f1))

    best_alpha = best_parameters['tfidf__alpha']
    best_beta  = best_parameters['tfidf__beta']
    rest_parameters = ', '.join([k+'='+str(best_parameters[k]) for k in parameters.keys() if k not in ['tfidf__alpha', 'tfidf__beta']])

    results.add_row(args.learner, args.dataset, vectorizer_name, nF, str(rest_parameters), best_alpha, best_beta, macro_f1, micro_f1)
