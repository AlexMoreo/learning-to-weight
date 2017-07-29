from __future__ import print_function

from pprint import pprint
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from data.dataset_loader import TextCollectionLoader

# class DataLoaderWrap:
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X, y=None):
#         return X
#     def fit_transform(self, X, y=None):
#         return X
#     def get_params(self,deep=True):
#         return {}
#     def set_params(self, dataset):
#         self.data = TextCollectionLoader(dataset=dataset)

pipeline = Pipeline([
    #('data', DataLoaderWrap()),
    # ('vect', CountVectorizer()),
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC()),
])




# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    #'data__dataset': ('reuters21578','20newsgroups'),
    #'vect__max_features': (None, 5000, 10000, 50000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__C': [10**i for i in range(0,2)],
    #'clf__n_iter': (10, 50, 80),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    #data = TextCollectionLoader(dataset='reuters21578', vectorizer='tf')
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    devel = fetch_20newsgroups(subset='train')
    grid_search.fit(devel.data, devel.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))