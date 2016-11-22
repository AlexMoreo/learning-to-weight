from sklearn import svm
import numpy as np
import time
from corpus_20newsgroup import *
from sklearn.metrics import *
from sklearn.preprocessing import normalize
import sys

for pos_cat_code in range(20):
    feat_sel = 10000
    categories = None #['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    data = Dataset(categories=categories, vectorize='tfidf', delete_metadata=True, dense=True, positive_cat=pos_cat_code, feat_sel=feat_sel)
    X,Y = data.get_devel_set()
    test, true_labels = data.test_batch()

    C = 1.0  # SVM regularization parameter
    #svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
    #rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
    #poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
    lin_svc = svm.LinearSVC(C=C).fit(X, Y)

    def evaluation(classifier):
        predictions = classifier.predict(test)
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='binary', pos_label=1)
        p = precision_score(true_labels, predictions, average='binary', pos_label=1)
        r = recall_score(true_labels, predictions, average='binary', pos_label=1)
        print('Cat %d: acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f' % (pos_cat_code, acc*100, f1, p, r))

    #evaluation(svc)
    #evaluation(rbf_svc)
    #evaluation(poly_svc)
    evaluation(lin_svc)

