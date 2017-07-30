from __future__ import print_function

from pprint import pprint
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import _document_frequency, TfidfTransformer
from data.custom_vectorizers import TfidfTransformerAlphaBeta
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2
import os
from sklearn.datasets import get_data_home
from data.dataset_loader import TextCollectionLoader
from utils.metrics import *

dataset = '20newsgroups'
min_df = 1 if dataset == 'reuters21578' else 3 #since reuters contains categories with less than 3 documents

pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english', min_df=min_df)),
    ('featsel', SelectKBest(chi2, k=2000)),
    ('tfidf', TfidfTransformerAlphaBeta(sublinear_tf=True, use_idf=True, norm='l2')),
    ('clf', LinearSVC()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'tfidf__alpha': [0.5],
    'tfidf__beta': [0.5,2.0],
    'clf__C': [10**i for i in range(0,2)],
}

if __name__ == "__main__":


    data_path = os.path.join(get_data_home(), dataset)
    if dataset == 'reuters21578':
        devel = TextCollectionLoader.fetch_reuters21579(data_path=data_path, subset='train')
        test = TextCollectionLoader.fetch_reuters21579(data_path=data_path, subset='test')
    elif dataset == '20newsgroups':
        devel = TextCollectionLoader.fetch_20newsgroups(data_path=data_path, subset='train')
        test = TextCollectionLoader.fetch_20newsgroups(data_path=data_path, subset='test')
    elif dataset == 'ohsumed20k':
        devel = TextCollectionLoader.fetch_ohsumed20k(data_path=data_path, subset='train')
        test = TextCollectionLoader.fetch_ohsumed20k(data_path=data_path, subset='test')

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()

    grid_search.fit(devel.data, devel.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print("Running test and evaluation:")
    y_ = grid_search.predict(test.data)
    macro_f1 = macroF1(test.target, y_)
    micro_f1 = microF1(test.target, y_)