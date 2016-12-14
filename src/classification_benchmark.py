import sys
import argparse
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
from dataset_loader import *
from weighted_vectors import WeightedVectors
from result_table import ReusltTable

#TODO: improve with GridSearchCV or RandomizedSearchCV

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
                        if best_f1 is None or f1 > best_f1:
                            best_f1 = f1
                            best_params = {'C':c, 'loss':l, 'dual':d, 'tol':tol}
                    except ValueError:
                        print 'Param configuration not supported, skip'

    results.init_row_result('LinearSVM', data)
    if isinstance(data, WeightedVectors):
        results.set_all(data.get_learning_parameters())

    if best_f1 is not None:
        print('Best params %s: f-score %f' % (str(best_params), best_f1))
        deX, deY = data.get_devel_set()
        teX, teY = data.get_test_set()
        svm_ = svm.LinearSVC(C=best_params['C'], loss=best_params['loss'], dual=best_params['dual'], tol=best_params['tol']).fit(deX, deY)
        teY_ = svm_.predict(teX)
        acc, f1, prec, rec = evaluation_metrics(predictions=teY_, true_labels=teY)
        print 'Test: acc=%.3f, f1=%.3f, p=%.3f, r=%.3f [pos=%d, truepos=%d]' % (acc, f1, prec, rec, sum(teY_), sum(teY))

        results.add_result_metric_scores(acc, f1, prec, rec, contingency_table(predictions=teY_, true_labels=teY),
                                         init_time,
                                         notes=str(best_params))

    else:
        results.set('notes', '<not applicable>')

    results.commit()

def random_forest(data, results):
    param_n_estimators = [10, 25, 50, 100]
    param_criterion = ['gini', 'entropy']
    param_max_features = ['sqrt', 'log2', 1000, None]
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
                        if best_f1 is None or f1 > best_f1:
                            best_f1 = f1
                            best_params = {'n_estimators':n_estimators, 'criterion':criterion, 'max_features':max_features, 'class_weight':class_weight}
                    except ValueError:
                        print 'Param configuration not supported, skip'

    results.init_row_result('RandomForest', data)
    if isinstance(data, WeightedVectors):
        results.set_all(data.get_learning_parameters())

    if best_f1 is not None:
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

        results.add_result_metric_scores(acc, f1, prec, rec, contingency_table(predictions=teY_, true_labels=teY), init_time,
                                         notes=str(best_params))

    else:
        results.set('notes', '<not applicable>')

    results.commit()

def multinomial_nb(data, results):
    param_alpha = [1.0, .1, .05, .01, .001, 0.0]
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
            if best_f1 is None or f1 > best_f1:
                best_f1 = f1
                best_params = {'alpha': alpha}
        except ValueError:
            print 'Param configuration not supported, skip'
        except IndexError:
            print 'Param configuration produced index error, skip'

    results.init_row_result('MultinomialNB', data)
    if isinstance(data, WeightedVectors):
        results.set_all(data.get_learning_parameters())

    if best_f1 is not None:
        print('Best params %s: f-score %f' % (str(best_params), best_f1))
        deX, deY = data.get_devel_set()
        teX, teY = data.get_test_set()
        nb_ = MultinomialNB(alpha=best_params['alpha']).fit(deX, deY)
        teY_ = nb_.predict(teX)
        acc, f1, prec, rec = evaluation_metrics(predictions=teY_, true_labels=teY)
        print 'Test: acc=%.3f, f1=%.3f, p=%.3f, r=%.3f' % (acc, f1, prec, rec)
        results.add_result_metric_scores(acc, f1, prec, rec, contingency_table(predictions=teY_, true_labels=teY),
                                         init_time,
                                         notes=str(best_params))
    else:
        results.set('notes', '<not applicable>')
    results.commit()

def logistic_regression(data, results):
    param_c = [1e4, 1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4]
    param_penalty = ['l2','l1']
    param_dual = [False, True]
    param_tol = [1e-4]
    trX, trY = data.get_train_set()
    vaX, vaY = data.get_validation_set()
    init_time = time.time()
    best_f1 = None
    for c in param_c:
        for l in param_penalty:
            for d in param_dual:
                for tol in param_tol:
                    try:
                        lr_ = LogisticRegression(C=c, penalty=l, dual=d, tol=tol, n_jobs=-1).fit(trX, trY)
                        vaY_ = lr_.predict(vaX)
                        _,f1,_,_=evaluation_metrics(predictions=vaY_, true_labels=vaY)
                        print 'Train Logistic Regression (c=%.3f, penalty=%s, dual=%s, tol=%f) got f-score=%f' % (c, l, d, tol, f1)
                        if best_f1 is None or f1 > best_f1:
                            best_f1 = f1
                            best_params = {'C':c, 'penalty':l, 'dual':d, 'tol':tol}
                    except ValueError:
                        print 'Param configuration not supported, skip'

    results.init_row_result('LogisticRegression', data)
    if isinstance(data, WeightedVectors):
        results.set_all(data.get_learning_parameters())

    if best_f1 is not None:
        print('Best params %s: f-score %f' % (str(best_params), best_f1))
        deX, deY = data.get_devel_set()
        teX, teY = data.get_test_set()
        lr_ = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], dual=best_params['dual'], tol=best_params['tol']).fit(deX, deY)
        teY_ = lr_.predict(teX)
        acc, f1, prec, rec = evaluation_metrics(predictions=teY_, true_labels=teY)
        print 'Test: acc=%.3f, f1=%.3f, p=%.3f, r=%.3f [pos=%d, truepos=%d]' % (acc, f1, prec, rec, sum(teY_), sum(teY))

        results.add_result_metric_scores(acc, f1, prec, rec, contingency_table(predictions=teY_, true_labels=teY),
                                         init_time,
                                         notes=str(best_params))

    else:
        results.set('notes', '<not applicable>')

    results.commit()

def run_benchmark(data, results, benchmarks):
    if benchmarks['linearsvm']:
        linear_svm(data, results)
    if benchmarks['multinomialnb']:
        multinomial_nb(data, results)
    if benchmarks['randomforest']:
        random_forest(data, results)
    if benchmarks['logisticregression']:
        logistic_regression(data, results)

if __name__ == '__main__':
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="indicates the dataset on which to run the baselines benchmark (ignored if --runbaselines False)", choices=['20newsgroups', 'reuters21578', 'movie_reviews'])
    parser.add_argument("-v", "--vectordir", help="directory containing learnt vectors in .pickle format", type=str)
    parser.add_argument("-r", "--resultfile", help="path to a result container file (.csv)", type=str, default="../results.csv")
    parser.add_argument("--no-linearsvm", help="removes the linearsvm classifier from the benchmark", default=False, action="store_true")
    parser.add_argument("--no-multinomialnb", help="removes the multinomialnb classifier from the benchmark", default=False, action="store_true")
    parser.add_argument("--no-randomforest", help="removes the randomforest classifier from the benchmark", default=False, action="store_true")
    parser.add_argument("--no-logisticregression", help="removes the logisticregression classifier from the benchmark", default=False, action="store_true")
    args = parser.parse_args()

    benchmarks = dict({'linearsvm': not args.no_linearsvm,
                       'multinomialnb': not args.no_multinomialnb,
                       'randomforest': not args.no_randomforest,
                       'logisticregression': not args.no_logisticregression})

    print "Loading result file from "+args.resultfile
    results = ReusltTable(args.resultfile)

    if args.dataset:
        print "Runing classification benchmark on baselines"
        print "Dataset: " + args.dataset
        if args.dataset == '20newsgroups':
            num_cats = 20
        elif args.dataset == 'reuters21578':
            num_cats = 115
        elif args.dataset == 'movie_reviews':
            num_cats = 2
        feat_sel = 10000
        for vectorizer in ['count', 'sublinear_tfidf', 'hashing', 'binary', 'tfidf']: #TODO tf, sublinear_tf, tf ig, bm25, l1...
            for pos_cat_code in range(num_cats):
                print('Category %d (%s)' % (pos_cat_code, vectorizer))
                data = DatasetLoader(dataset=args.dataset, vectorize=vectorizer, rep_mode='sparse', positive_cat=pos_cat_code, feat_sel=feat_sel)
                print("|Tr|=%d [prev+ %f]" % (data.num_tr_documents(), data.train_class_prevalence()))
                print("|Val|=%d [prev+ %f]" % (data.num_val_documents(), data.valid_class_prevalence()))
                print("|Te|=%d [prev+ %f]" % (data.num_test_documents(), data.test_class_prevalence()))

                run_benchmark(data, results, benchmarks)
    if args.vectordir:
        print "Runing classification benchmark on learnt vectors in " + args.vectordir
        for vecname in [pickle for pickle in os.listdir(args.vectordir) if pickle.endswith('.pickle')]:
            print "Vector file: " + vecname
            data = WeightedVectors.unpickle(indir=args.vectordir, infile_name=vecname)
            run_benchmark(data, results, benchmarks)

    print "Done."
    results.commit()



