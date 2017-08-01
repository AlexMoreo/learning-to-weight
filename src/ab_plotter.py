from __future__ import print_function
from pprint import pprint
from time import time
from custom_vectorizers import TfidfTransformerAlphaBeta, BM25TransformerAlphaBeta, TSRweightingAlphaBeta
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from dataset_loader import TextCollectionLoader
from sklearn.multiclass import OneVsRestClassifier
from tsr_function import information_gain, chi_square, relevance_frequency, conf_weight, gain_ratio, get_supervised_matrix
from metrics import *
import sys,os
from os.path import join
import pandas as pd
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt

if 'MATPLOTLIB_USE' in os.environ:
    matplotlib.use(os.environ['MATPLOTLIB_USE'])
import matplotlib.pyplot as plt
from matplotlib import cm


def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

def plot3D(x, y, z, title='', zlabel='', save_to_pdf=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, linewidth=0.2, cmap=cm.jet)
    if title:
        ax.set_title(title)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    if zlabel:
        ax.set_zlabel(zlabel)
    if save_to_pdf:
        dir = os.path.dirname(save_to_pdf)
        if dir and not os.path.exists(dir): os.makedirs(dir)
        plt.savefig(save_to_pdf, format='pdf')
    else:
        plt.show()

if __name__ == "__main__":

    supervised_weight_functions = ['tfidf', 'bm25', 'tfig', 'tfgr', 'tfchi', 'tfrf', 'tfcw']
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="indicates the dataset on which to run the baselines benchmark ", choices=TextCollectionLoader.valid_datasets)
    parser.add_argument("-r", "--outdir", help="directory where to save the plots", type=str, default="../plots")
    parser.add_argument("-v", "--vectorizer", help="selects one single vectorizer method to run from", type=str, choices=supervised_weight_functions)
    parser.add_argument("--fs", help="feature selection ratio", type=float, default=0.1)
    parser.add_argument("--sublinear_tf", help="logarithmic version of the tf-like function", default=False, action="store_true")
    parser.add_argument("--global_policy", help="global policy for supervised term weighting approaches", choices=['max', 'ave', 'wave', 'sum'], type=str, default='max')
    parser.add_argument("-l", "--learner", help="learner", type=str, default='LinearSVC')
    args = parser.parse_args()

    data = TextCollectionLoader(dataset=args.dataset, vectorizer='tf', rep_mode='sparse', feat_sel=args.fs, sublinear_tf=False)
    nF = data.num_features()

    Xtr, ytr = data.get_devel_set()
    Xte, yte = data.get_test_set()

    if args.learner == 'LinearSVC':
        clf = OneVsRestClassifier(LinearSVC())
        clf_params = {'estimator__C': [10 ** i for i in range(0, 3)]}
    else:
        print("Learner {} is not supported. Exit.".format(args.learner))

    vectorizer_name = ('Log' if args.sublinear_tf else '') + args.vectorizer

    print("Init exploration")
    macro_results = []
    micro_results = []
    grid = np.linspace(0.25, 2.0, 16)
    for alpha in grid:
        for beta in grid:

            tsr_map = {'tfig': information_gain, 'tfchi': chi_square, 'tfgr': gain_ratio, 'tfrf': relevance_frequency, 'tfcw': conf_weight}
            if args.vectorizer == 'tfidf':
                vect = TfidfTransformerAlphaBeta(alpha=alpha, beta=beta, sublinear_tf=args.sublinear_tf, use_idf=True, norm='l2')
            elif args.vectorizer == 'bm25':
                vect = BM25TransformerAlphaBeta(alpha=alpha, beta=beta, norm='none')
            elif args.vectorizer in supervised_weight_functions:
                supervised_matrix = get_supervised_matrix(Xtr, ytr)
                vect = TSRweightingAlphaBeta(alpha=alpha, beta=beta, tsr_function=tsr_map[args.vectorizer], global_policy=args.global_policy, sublinear_tf=args.sublinear_tf, supervised_4cell_matrix=supervised_matrix)
            else:
                print("Vectorizer {} is not supported. Exit.".format(args.vectorizer))

            wXtr = vect.fit_transform(Xtr,ytr)
            wXte = vect.transform(Xte)

            t0 = time()
            grid_search = GridSearchCV(clf, clf_params, n_jobs=-1, verbose=1)
            print("Performing grid search...")
            print("parameters:")
            pprint(clf_params)
            grid_search.fit(wXtr, ytr)
            print("done in %0.3fs" % (time() - t0))
            print()

            print("Best score: %0.3f" % grid_search.best_score_)
            print("Best parameters set:")
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(clf_params.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))

            print("Running test and evaluation:")
            y_ = grid_search.predict(wXte)
            macro_f1 = macroF1(yte, y_)
            micro_f1 = microF1(yte, y_)
            print("A={} B={} MacroF1={}, microF1={}".format(alpha, beta, macro_f1, micro_f1))

            macro_results.append((alpha, beta, macro_f1))
            micro_results.append((alpha, beta, micro_f1))

    out_path = join(args.outdir, args.dataset, vectorizer_name)

    x, y, z = zip(*macro_results)
    plot3D(x, y, z, zlabel=r'$F_1^M$', save_to_pdf=out_path + '_'+args.learner+'_macro_f1.pdf')

    x, y, z = zip(*micro_results)
    plot3D(x, y, z, zlabel=r'$F_1^\mu$', save_to_pdf=out_path + '_'+args.learner+ '_micro_f1.pdf')
