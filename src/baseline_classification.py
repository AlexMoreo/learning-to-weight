from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
from time import gmtime, strftime
from corpus_20newsgroup import *
from sklearn.metrics import *
from sklearn.preprocessing import normalize
import sys
import pandas as pd

def init_row_result(classifier_name, data, run=0):
    results.add_empty_entry()
    results.set('classifier', classifier_name)
    results.set('vectorizer', data.vectorize)
    results.set('num_features', data.num_features())
    results.set('dataset', data.name)
    results.set('category', data.positive_cat)
    results.set('run', run)

def add_result_metric_scores(best_params, acc, f1, prec, rec, cont_table, init_time):
    results.set('notes', str(best_params))
    results.set_all({'acc': acc, 'fscore': f1, 'precision': prec, 'recall': rec})
    results.set_all(cont_table)
    results.set_all({'date': strftime("%d-%m-%Y", gmtime()), 'time': init_time, 'elapsedtime': time.time() - init_time})

def linear_svm(data, results):
    param_c = [1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3]
    param_loss = ['hinge','squared_hinge']
    param_dual = [False, True]
    param_tol = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    trX, trY = data.get_train_set()
    vaX, vaY = data.get_validation_set()
    init_time = time.time()
    best_f1 = None
    for c in param_c:
        for l in param_loss:
            for d in param_dual:
                for tol in param_tol:
                    try:
                        svm_ = svm.LinearSVC(C=c, loss=l, dual=d, tol=tol).fit(trX, trY)
                        vaY_ = svm_.predict(vaX)
                        _,f1,_,_=evaluation_metrics(predictions=vaY_, true_labels=vaY)
                        print 'Train SVM (c=%.3f, loss=%s, dual=%s, tol=%f) got f-score=%f' % (c, l, d, tol, f1)
                        if not best_f1 or f1 > best_f1:
                            best_f1 = f1
                            best_params = {'C':c, 'loss':l, 'dual':d, 'tol':tol}
                    except ValueError:
                        print 'Param configuration not supported, skip'

    init_row_result('LinearSVM', data)

    if best_f1:
        print('Best params %s: f-score %f' % (str(best_params), best_f1))
        deX, deY = data.get_devel_set()
        teX, teY = data.get_test_set()
        svm_ = svm.LinearSVC(C=best_params['C'], loss=best_params['loss'], dual=best_params['dual'], tol=best_params['tol']).fit(deX, deY)
        teY_ = svm_.predict(teX)
        acc, f1, prec, rec = evaluation_metrics(predictions=teY_, true_labels=teY)
        print 'Test: acc=%.3f, f1=%.3f, p=%.3f, r=%.3f' % (acc, f1, prec, rec)

        add_result_metric_scores(best_params, acc, f1, prec, rec, contingency_table(predictions=teY_, true_labels=teY), init_time)

    else:
        results.set('notes', '<not applicable>')

    results.commit()



def random_forest(data, results):
    param_n_estimators = [10, 25, 50, 100]
    param_criterion = ['gini', 'entropy']
    param_max_features = ['auto', 'sqrt', 'log2', None]
    param_class_weight = ['balanced', 'balanced_subsample', None]
    trX, trY = data.get_train_set()
    vaX, vaY = data.get_validation_set()
    best_f1 = None
    init_time = time.time()
    for n_estimators in param_n_estimators:
        for criterion in param_criterion:
            for max_features in param_max_features:
                for class_weight in param_class_weight:
                    try:
                        rf_ = RandomForestClassifier(n_estimators=n_estimators,
                            criterion=criterion, max_features=max_features, class_weight=class_weight, n_jobs=-1).fit(trX, trY)
                        vaY_ = rf_.predict(vaX)
                        _, f1, _, _ = evaluation_metrics(predictions=vaY_, true_labels=vaY)
                        print 'Train Random Forest (n_estimators=%.3f, criterion=%s, max_features=%s, class_weight=%s) got f-score=%f' % \
                              (n_estimators, criterion, max_features, class_weight, f1)
                        if not best_f1 or f1 > best_f1:
                            best_f1 = f1
                            best_params = {'n_estimators':n_estimators, 'criterion':criterion, 'max_features':max_features, 'class_weight':class_weight}
                    except ValueError:
                        print 'Param configuration not supported, skip'

    init_row_result('RandomForest', data)

    if best_f1:
        print('Best params %s: f-score %f' % (str(best_params), best_f1))
        deX, deY = data.get_devel_set()
        teX, teY = data.get_test_set()
        rf_ = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                     criterion=best_params['criterion'],
                                     max_features=best_params['max_features'],
                                     class_weight=best_params['class_weight'],
                                     n_jobs=-1).fit(deX, deY)
        teY_ = rf_.predict(teX)
        acc, f1, prec, rec = evaluation_metrics(predictions=teY_, true_labels=teY)
        print 'Test: acc=%.3f, f1=%.3f, p=%.3f, r=%.3f' % (acc, f1, prec, rec)

        add_result_metric_scores(best_params, acc, f1, prec, rec, contingency_table(predictions=teY_, true_labels=teY), init_time)

    else:
        results.set('notes', '<not applicable>')

    results.commit()

def multinomial_nb(data, results):
    param_alpha = [1.0, .1, .01, .001, 0.0]
    trX, trY = data.get_train_set()
    vaX, vaY = data.get_validation_set()
    best_f1 = None
    init_time = time.time()
    for alpha in param_alpha:
        try:
            nb_ = MultinomialNB(alpha=alpha).fit(trX, trY)
            vaY_ = nb_.predict(vaX)
            _, f1, _, _ = evaluation_metrics(predictions=vaY_, true_labels=vaY)
            print 'Train Multinomial (alpha=%.3f) got f-score=%f' % (alpha, f1)
            if not best_f1 or f1 > best_f1:
                best_f1 = f1
                best_params = {'alpha': alpha}
        except ValueError:
            print 'Param configuration not supported, skip'

    init_row_result('MultinomialNB', data)

    if best_f1:
        print('Best params %s: f-score %f' % (str(best_params), best_f1))
        deX, deY = data.get_devel_set()
        teX, teY = data.get_test_set()
        nb_ = MultinomialNB(alpha=best_params['alpha']).fit(deX, deY)
        teY_ = nb_.predict(teX)
        acc, f1, prec, rec = evaluation_metrics(predictions=teY_, true_labels=teY)
        print 'Test: acc=%.3f, f1=%.3f, p=%.3f, r=%.3f' % (acc, f1, prec, rec)
        add_result_metric_scores(best_params, acc, f1, prec, rec, contingency_table(predictions=teY_, true_labels=teY), init_time)
    else:
        results.set('notes', '<not applicable>')
    results.commit()


def evaluation_metrics(predictions, true_labels):
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='binary', pos_label=1)
    p = precision_score(true_labels, predictions, average='binary', pos_label=1)
    r = recall_score(true_labels, predictions, average='binary', pos_label=1)
    return acc, f1, p, r

def contingency_table(predictions, true_labels):
    t = confusion_matrix(true_labels, predictions)
    return {'tp':t[1, 1], 'tn':t[0, 0], 'fn':t[1,0], 'fp':t[0,1]}

class ReusltTable:
    def __init__(self, result_container):
        self.result_container = result_container

        if os.path.exists(result_container):
            self.df = pd.read_csv(result_container)
        else:
            self.df = pd.DataFrame(columns=['classifier',  # linearsvm, random forest, Multinomial NB,
                                       'vectorizer',  # binary, count, tf, tfidf, tfidf sublinear, bm25, hashing, learnt
                                       'num_features',
                                       'dataset',  # 20newsgroup, rcv1, ...
                                       'category',
                                       'run',
                                       'date',
                                       'time',
                                       'elapsedtime',
                                       'hiddensize',
                                       'lrate',
                                       'optimizer',
                                       'normalize',
                                       'nonnegative',
                                       'pretrain',
                                       'iterations',
                                       'notes',
                                       'acc', 'fscore', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn'])

    def add_empty_entry(self):
        self.df.loc[len(self.df)] = [np.nan] * len(self.df.columns)

    def set(self, column, value):
        #self.df.loc[len(self.df-1)][list(self.df.columns).index(column)] = value
        self.df.iloc[len(self.df) - 1, list(self.df.columns).index(column)] = value

    def set_all(self, dictionary):
        for key,value in dictionary.items():
            self.set(key,value)

    def commit(self):
        self.df.to_csv(self.result_container, index=False)

if __name__ == '__main__':
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    #TODO parse params (num categories, feat sel, vectorizer (count, tf, tfidf, tfidf sublinear, bm25, hashing, binary, tf ig, tf others...), methods, read-from-model)
    # puedo cambuar la score_function en SelectKBest

    result_container = '../results.csv'
    results = ReusltTable(result_container)

    num_cats = 20

    for vectorizer in ['hashing', 'binary','count','tfidf','sublinear_tfidf']:
        for pos_cat_code in range(num_cats):
            print('Category %d (%s)' % (pos_cat_code, vectorizer))


            data = Dataset(categories=None, vectorize='binary', delete_metadata=True, rep_mode='sparse',
                           positive_cat=15, feat_sel=10000)
            linear_svm(data, results)
            sys.exit()

            feat_sel = 10000
            categories = None #['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
            data = Dataset(categories=categories, vectorize=vectorizer, delete_metadata=True, dense=True, positive_cat=pos_cat_code, feat_sel=feat_sel)

            linear_svm(data, results)
            multinomial_nb(data, results)
            random_forest(data, results)

    print results.df
    results.commit()



