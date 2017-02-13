from __future__ import print_function

import argparse

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from src.data.dataset_loader import *
from src.data.weighted_vectors import WeightedVectors
from src.utils.result_table import ResultTable

#TODO: improve with GridSearchCV or RandomizedSearchCV

n_jobs = -1

def featsel(trX, trY, teX, n_feat):
    print('Selecting top-%d features from %d...'%(n_feat, trX.shape[1]))
    fs = SelectKBest(chi2, k=n_feat)
    trX_red = fs.fit_transform(trX, trY)
    teX_red = fs.transform(teX)
    return trX_red, teX_red

def knn(data, results):
    t_ini = time.time()
    param_k = [15,5,3,1]
    param_weight = ['distance']#['distance', 'uniform']
    param_pca = [None, 64, 128]
    feat_sel  = [None, 500, 250, 100, 50, 25]
    trX, trY = data.get_train_set()
    vaX, vaY = data.get_validation_set()

    tr_positive_examples = sum(trY)
    init_time = time.time()
    best_f1 = None
    for fs in feat_sel:
        if fs is not None:
            trX, vaX = featsel(trX, trY, vaX, fs)
        for pca_components in param_pca:
            if best_f1 == 1.0: break
            if pca_components is not None:
                if fs is not None: continue
                if data.vectorize=='hashing': continue
                if pca_components >= trX.shape[1]: continue
                print("PCA(%s) from %d dimensions" % (pca_components, trX.shape[1]))
                pca = PCA(n_components=pca_components)
                trX_pca = pca.fit_transform(trX.todense())
                vaX_pca = pca.transform(vaX.todense())
            else:
                trX.sort_indices()
                vaX.sort_indices()
                #trX = sklearn.preprocessing.normalize(trX, norm='l2', axis=1, copy=False)
                #vaX = sklearn.preprocessing.normalize(vaX, norm='l2', axis=1, copy=False)
                trX_pca, vaX_pca = trX, vaX

            for k in param_k:
                if k > tr_positive_examples: continue
                for w in param_weight:
                    if k==1 and w=='uniform': continue
                    try:
                        if best_f1 == 1.0: break
                        knn_ = KNeighborsClassifier(n_neighbors=k, weights=w, n_jobs=n_jobs).fit(trX_pca, trY)
                        vaY_ = knn_.predict(vaX_pca)
                        _,f1,_,_=evaluation_metrics(predictions=vaY_, true_labels=vaY)
                        print('Train KNN (fs=%s, pca=%s, k=%d, weights=%s) got f-score=%f' % (fs, pca_components, k, w, f1))
                        if best_f1 is None or f1 > best_f1:
                            best_f1 = f1
                            best_params = {'k':k, 'w':w, 'fs':fs, 'pca':pca_components}
                            #print('\rTrain KNN (pca=%d, k=%d, weights=%s) got f-score=%f' % (pca_components if pca_components is not None else data.num_features(), k, w, f1), end='')

                    except ValueError:
                        pass #print('Param configuration not supported, skip')

    results.init_row_result('KNN', data)
    if isinstance(data, WeightedVectors):
        results.set_all(data.get_learning_parameters())

    if best_f1 is not None:
        print('\nBest params %s: f-score %f' % (str(best_params), best_f1))
        deX, deY = data.get_devel_set()
        teX, teY = data.get_test_set()
        #sorting indexes is a work-around for a parallel issue due to n_jobs!=1 and in-place internal assignments

        if best_params['fs'] is not None:
            deX, teX = featsel(deX, deY, teX, best_params['fs'])
        if best_params['pca'] is not None:
            pca = PCA(n_components=best_params['pca'])
            deX_pca = pca.fit_transform(deX.todense())
            teX_pca = pca.transform(teX.todense())
        else:
            deX.sort_indices()
            teX.sort_indices()
            deX_pca, teX_pca = deX, teX

        knn_ = KNeighborsClassifier(n_neighbors=best_params['k'], weights=best_params['w'], n_jobs=n_jobs).fit(deX_pca, deY)
        teY_ = knn_.predict(teX_pca)
        acc, f1, prec, rec = evaluation_metrics(predictions=teY_, true_labels=teY)
        print('Test: acc=%.3f, f1=%.3f, p=%.3f, r=%.3f [pos=%d, truepos=%d] took %.3fsec.\n' % (acc, f1, prec, rec, sum(teY_), sum(teY), time.time()-t_ini))

        results.add_result_metric_scores(acc, f1, prec, rec, contingency_table(predictions=teY_, true_labels=teY),
                                         init_time,
                                         notes=str(best_params))

    else:
        results.set('notes', '<not applicable>')

    results.commit()

def linear_svm(data, results):
    param_c = [1e4, 1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4]
    param_loss = ['hinge','squared_hinge']
    param_dual = [False, True]
    trX, trY = data.get_train_set()
    vaX, vaY = data.get_validation_set()
    init_time = time.time()
    best_f1 = None
    for c in param_c:
        for l in param_loss:
            for d in param_dual:
                try:
                    svm_ = svm.LinearSVC(C=c, loss=l, dual=d).fit(trX, trY)
                    vaY_ = svm_.predict(vaX)
                    _,f1,_,_=evaluation_metrics(predictions=vaY_, true_labels=vaY)
                    #print('Train SVM (c=%.3f, loss=%s, dual=%s) got f-score=%f' % (c, l, d, f1))
                    if best_f1 is None or f1 > best_f1:
                        best_f1 = f1
                        best_params = {'C':c, 'loss':l, 'dual':d}
                        print('\rTrain SVM (c=%.3f, loss=%s, dual=%s) got f-score=%f' % (c, l, d, f1), end='')
                except ValueError:
                    pass #print('Param configuration not supported, skip')

    results.init_row_result('LinearSVM', data)
    if isinstance(data, WeightedVectors):
        results.set_all(data.get_learning_parameters())

    if best_f1 is not None:
        print('\nBest params %s: f-score %f' % (str(best_params), best_f1))
        deX, deY = data.get_devel_set()
        teX, teY = data.get_test_set()
        svm_ = svm.LinearSVC(C=best_params['C'], loss=best_params['loss'], dual=best_params['dual']).fit(deX, deY)
        teY_ = svm_.predict(teX)
        acc, f1, prec, rec = evaluation_metrics(predictions=teY_, true_labels=teY)
        print('Test: acc=%.3f, f1=%.3f, p=%.3f, r=%.3f [pos=%d, truepos=%d]\n' % (acc, f1, prec, rec, sum(teY_), sum(teY)))

        results.add_result_metric_scores(acc, f1, prec, rec, contingency_table(predictions=teY_, true_labels=teY),
                                         init_time,
                                         notes=str(best_params))

    else:
        results.set('notes', '<not applicable>')

    results.commit()

def random_forest(data, results):
    param_n_estimators = [10, 25, 50, 100]
    param_criterion = ['gini', 'entropy']
    param_max_features = ['sqrt', 'log2', 1000] #The None configuration (all) is extremely slow
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
                            criterion=criterion, max_features=max_features, class_weight=class_weight, n_jobs=n_jobs).fit(trX, trY)
                        vaY_ = rf_.predict(vaX)
                        _, f1, _, _ = evaluation_metrics(predictions=vaY_, true_labels=vaY)
                        #print('Train Random Forest (n_estimators=%.3f, criterion=%s, max_features=%s, class_weight=%s) got f-score=%f' % \
                        #      (n_estimators, criterion, max_features, class_weight, f1))
                        if best_f1 is None or f1 > best_f1:
                            best_f1 = f1
                            best_params = {'n_estimators':n_estimators, 'criterion':criterion, 'max_features':max_features, 'class_weight':class_weight}
                            print('\rTrain Random Forest (n_estimators=%d, criterion=%s, max_features=%s, class_weight=%s) got f-score=%f' % \
                              (n_estimators, criterion, max_features, class_weight, f1), end='')
                    except ValueError:
                        pass #print('Param configuration not supported, skip')

    results.init_row_result('RandomForest', data)
    if isinstance(data, WeightedVectors):
        results.set_all(data.get_learning_parameters())

    if best_f1 is not None:
        print('\nBest params %s: f-score %f' % (str(best_params), best_f1))
        deX, deY = data.get_devel_set()
        teX, teY = data.get_test_set()
        rf_ = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                     criterion=best_params['criterion'],
                                     max_features=best_params['max_features'],
                                     class_weight=best_params['class_weight'],
                                     n_jobs=n_jobs).fit(deX, deY)
        teY_ = rf_.predict(teX)
        acc, f1, prec, rec = evaluation_metrics(predictions=teY_, true_labels=teY)
        print('Test: acc=%.3f, f1=%.3f, p=%.3f, r=%.3f\n' % (acc, f1, prec, rec))

        results.add_result_metric_scores(acc, f1, prec, rec, contingency_table(predictions=teY_, true_labels=teY), init_time,
                                         notes=str(best_params))

    else:
        results.set('notes', '<not applicable>')

    results.commit()

def multinomial_nb(data, results):
    def swap_vectors_sign(data):
        if data.vectorize != 'learned': return
        def mainly_negative_nonzeros(csr_m):
            values = np.array(csr_m[csr_m.nonzero()])[0]
            negatives = len(values[values<0])
            if negatives == 0: return False
            positives = len(values[values>0])
            return negatives > positives
        def swap_sign(csr_m):
            return csr_m.multiply(-1)
        def del_negatives(swapped):
            swapped[swapped<0]=0
            swapped.eliminate_zeros()
            return swapped
        if mainly_negative_nonzeros(data.trX):
            data.trX = swap_sign(data.trX)
            data.vaX = swap_sign(data.vaX)
            data.teX = swap_sign(data.teX)
        data.trX = del_negatives(data.trX)
        data.vaX = del_negatives(data.vaX)
        data.teX = del_negatives(data.teX)

    #if all vectors are non-positive, swaps their sign -- otherwise the multinomial nb could not be computed.
    swap_vectors_sign(data)

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
            print('Train Multinomial (alpha=%.3f) got f-score=%f' % (alpha, f1))
            if best_f1 is None or f1 > best_f1:
                best_f1 = f1
                best_params = {'alpha': alpha}
        except ValueError:
            print('Param configuration not supported, skip')
        except IndexError:
            print('Param configuration produced index error, skip')

    results.init_row_result('MultinomialNB', data)
    if isinstance(data, WeightedVectors):
        results.set_all(data.get_learning_parameters())

    if best_f1 is not None:
        print('\nBest params %s: f-score %f' % (str(best_params), best_f1))
        deX, deY = data.get_devel_set()
        teX, teY = data.get_test_set()
        nb_ = MultinomialNB(alpha=best_params['alpha']).fit(deX, deY)
        teY_ = nb_.predict(teX)
        acc, f1, prec, rec = evaluation_metrics(predictions=teY_, true_labels=teY)
        print('Test: acc=%.3f, f1=%.3f, p=%.3f, r=%.3f\n' % (acc, f1, prec, rec))
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
    trX, trY = data.get_train_set()
    vaX, vaY = data.get_validation_set()
    init_time = time.time()
    best_f1 = None
    for c in param_c:
        for l in param_penalty:
            for d in param_dual:
                try:
                    lr_ = LogisticRegression(C=c, penalty=l, dual=d, n_jobs=n_jobs).fit(trX, trY)
                    vaY_ = lr_.predict(vaX)
                    _,f1,_,_=evaluation_metrics(predictions=vaY_, true_labels=vaY)
                    #print('Train Logistic Regression (c=%.3f, penalty=%s, dual=%s) got f-score=%f' % (c, l, d, f1))
                    if best_f1 is None or f1 > best_f1:
                        best_f1 = f1
                        best_params = {'C':c, 'penalty':l, 'dual':d}
                        print('Train Logistic Regression (c=%.3f, penalty=%s, dual=%s) got f-score=%f' % (c, l, d, f1), end="")
                except ValueError:
                    pass #print('Param configuration not supported, skip')

    results.init_row_result('LogisticRegression', data)
    if isinstance(data, WeightedVectors):
        results.set_all(data.get_learning_parameters())

    if best_f1 is not None:
        print('\nBest params %s: f-score %f' % (str(best_params), best_f1))
        deX, deY = data.get_devel_set()
        teX, teY = data.get_test_set()
        lr_ = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], dual=best_params['dual']).fit(deX, deY)
        teY_ = lr_.predict(teX)
        acc, f1, prec, rec = evaluation_metrics(predictions=teY_, true_labels=teY)
        print('Test: acc=%.3f, f1=%.3f, p=%.3f, r=%.3f [pos=%d, truepos=%d]\n' % (acc, f1, prec, rec, sum(teY_), sum(teY)))

        results.add_result_metric_scores(acc, f1, prec, rec, contingency_table(predictions=teY_, true_labels=teY),
                                         init_time,
                                         notes=str(best_params))

    else:
        results.set('notes', '<not applicable>')

    results.commit()

def run_benchmark(data, results, benchmarks):
    if benchmarks['linearsvm']:
        linear_svm(data, results)
    if benchmarks['randomforest']:
        random_forest(data, results)
    if benchmarks['logisticregression']:
        logistic_regression(data, results)
    if benchmarks['multinomialnb']:
        multinomial_nb(data, results)
    if benchmarks['knn']:
        knn(data, results)

if __name__ == '__main__':
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="indicates the dataset on which to run the baselines benchmark ", choices=DatasetLoader.valid_datasets)
    parser.add_argument("-v", "--vectordir", help="directory containing learnt vectors in .pickle format", type=str)
    parser.add_argument("-r", "--resultfile", help="path to a result container file (.csv)", type=str, default="../results.csv")
    parser.add_argument("-m", "--method", help="selects one single vectorizer method to run from "+str(DatasetLoader.valid_vectorizers),
                        type=str, default="all")
    parser.add_argument("--fs", help="feature selection ratio", type=float, default=0.1)
    parser.add_argument("--no-linearsvm", help="removes the linearsvm classifier from the benchmark", default=False, action="store_true")
    parser.add_argument("--no-multinomialnb", help="removes the multinomialnb classifier from the benchmark", default=False, action="store_true")
    parser.add_argument("--no-randomforest", help="removes the randomforest classifier from the benchmark", default=False, action="store_true")
    parser.add_argument("--no-logisticregression", help="removes the logisticregression classifier from the benchmark", default=False, action="store_true")
    parser.add_argument("--no-knn", help="removes the knn classifier from the benchmark", default=False, action="store_true")
    args = parser.parse_args()

    benchmarks = dict({'linearsvm': not args.no_linearsvm,
                       'multinomialnb': not args.no_multinomialnb,
                       'randomforest': not args.no_randomforest,
                       'logisticregression': not args.no_logisticregression,
                       'knn': not args.no_knn})

    print("Loading result file from "+args.resultfile)
    results = ResultTable(args.resultfile)

    if args.dataset:
        print("Runing classification benchmark on baselines")
        print("Dataset: " + args.dataset)
        feat_sel = args.fs
        for vectorizer in ([args.method] if args.method!='all' else DatasetLoader.valid_vectorizers):
            for pos_cat_code in DatasetLoader.valid_catcodes[args.dataset]:
                print('Category %d (%s)' % (pos_cat_code, vectorizer))
                data = DatasetLoader(dataset=args.dataset, vectorize=vectorizer, rep_mode='sparse', positive_cat=pos_cat_code, feat_sel=feat_sel)
                print("|Tr|=%d [prev+ %f]" % (data.num_tr_documents(), data.train_class_prevalence()))
                print("|Val|=%d [prev+ %f]" % (data.num_val_documents(), data.valid_class_prevalence()))
                print("|Te|=%d [prev+ %f]" % (data.num_test_documents(), data.test_class_prevalence()))

                run_benchmark(data, results, benchmarks)
    if args.vectordir:
        print("Runing classification benchmark on learnt vectors in " + args.vectordir)
        vectors = [pickle for pickle in os.listdir(args.vectordir) if pickle.endswith('.pickle')]
        for i,vecname in enumerate(vectors):
            print("Vector file: " + vecname)
            data = WeightedVectors.unpickle(indir=args.vectordir, infile_name=vecname)
            run_benchmark(data, results, benchmarks)
            print("Completed %d/%d" % (i,len(vectors)))
            print('-'*80)

    print("Done.")
    results.commit()



