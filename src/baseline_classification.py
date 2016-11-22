from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import time
from corpus_20newsgroup import *
from sklearn.metrics import *
from sklearn.preprocessing import normalize
import sys

def train_classifiers(trX, trY, teX, teY):
    C = 1.0  # SVM regularization parameter
    # svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
    #print 'Training Gaussian SVM'
    #rbf_svc = svm.SVC(kernel='rbf', C=C).fit(X, Y)
    evaluation(svm.LinearSVC(C=C).fit(trX, trY), teX, teY, 'Lin-SVM')
    evaluation(svm.SVC(kernel='poly', C=C).fit(trX, trY), teX, teY, 'Poly-SVM')
    evaluation(KNeighborsClassifier(n_jobs=-1).fit(trX, trY), teX, teY, 'k-NN')
    evaluation(DecisionTreeClassifier().fit(trX, trY), teX, teY, 'Decision Tree')
    evaluation(MultinomialNB().fit(trX, trY), teX, teY, 'MultiNB')

def evaluation(classifier, test, true_labels, name=''):
    predictions = classifier.predict(test)
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='binary', pos_label=1)
    p = precision_score(true_labels, predictions, average='binary', pos_label=1)
    r = recall_score(true_labels, predictions, average='binary', pos_label=1)
    print(name + ': acc=%.3f%%, f1=%.3f, p=%.3f, r=%.3f' % (acc * 100, f1, p, r))

if __name__ == '__main__':

    #TODO parse params (num categories, feat sel, vectorizer, methods)
    for pos_cat_code in range(20):
        feat_sel = 10000
        categories = None #['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
        data = Dataset(categories=categories, vectorize='tfidf', delete_metadata=True, dense=True, positive_cat=pos_cat_code, feat_sel=feat_sel)
        trX, trY = data.get_devel_set()
        teX, teY = data.test_batch()

        train_classifiers(trX, trY, teX, teY)





