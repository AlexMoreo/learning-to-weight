import argparse

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from utils.metrics import *


from data.dataset_loader import *
from data.weighted_vectors import WeightedVectors
from utils.result_table import BaselineResultTable, Learning2Weight_ResultTable

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

def fit_model_hyperparameters(data, parameters, model):
    single_class = data.num_categories() == 1
    if not single_class:
        parameters = {'estimator__' + key: parameters[key] for key in parameters.keys()}
        model = OneVsRestClassifier(model, n_jobs=-1)
    model_tunning = GridSearchCV(model, param_grid=parameters,
                                 scoring=make_scorer(macroF1), error_score=0, refit=True, cv=5, n_jobs=-1)

    Xtr, ytr = data.get_devel_set()
    Xtr.sort_indices()
    if single_class:
        ytr = np.squeeze(ytr)

    tunned = model_tunning.fit(Xtr, ytr)

    return tunned


def fit_and_test_model(data, parameters, model, results):
    single_class = data.num_categories() == 1
    if results.check_if_calculated(classifier=type(model).__name__,
                                weighting=data.vectorizer_name,
                                num_features=data.num_features(),
                                dataset=data.name,
                                category=data.positive_cat if single_class else 'all'):
        print "Skip already calculated experiment."
        return

    init_time = time.time()
    tunned_model = fit_model_hyperparameters(data, parameters, model)
    tunning_time = time.time() - init_time
    print("%s: best parameters %s, best score %.3f, took %.3f seconds" %
          (type(model).__name__, tunned_model.best_params_, tunned_model.best_score_, tunning_time))

    Xte, yte = data.get_test_set()
    Xte.sort_indices()
    yte_ = tunned_model.predict(Xte)

    results.init_row_result(type(model).__name__, data)
    if isinstance(data, WeightedVectors):
        results.set_all(data.get_learning_parameters())

    if single_class:
        result_4cell_table = single_metric_statistics(np.squeeze(yte), yte_)
        fscore = f1(result_4cell_table)
        acc = accuracy(result_4cell_table)
        results.add_result_scores_binary(acc, fscore, result_4cell_table, init_time, notes=tunned_model.best_params_)
        print("Test scores: %.3f acc, %.3f f1" % (acc, fscore))
    else:
        macro_f1 = macroF1(yte, yte_)
        micro_f1 = microF1(yte, yte_)
        results.add_result_scores_multiclass(macro_f1, micro_f1, init_time, notes=tunned_model.best_params_)
        print("Test scores: %.3f macro-f1, %.3f micro-f1" % (macro_f1, micro_f1))
    results.commit()


def linear_svm(data, results):
    parameters = {'C': [1e4, 1e3, 1e2, 1e1, 1],
                  'loss': ['hinge', 'squared_hinge'],
                  'dual': [True, False]}
    model = LinearSVC()
    fit_and_test_model(data, parameters, model, results)


def random_forest(data, results):
    parameters = {'n_estimators': [10, 25, 50, 100],
                  'criterion': ['gini', 'entropy'],
                  'max_features': ['sqrt', 'log2', 1000], #The None configuration (all) is extremely slow
                  'class_weight': ['balanced', 'balanced_subsample', None]
                 }
    model = RandomForestClassifier() #check multiclass config
    fit_and_test_model(data, parameters, model, results)

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

    parameters = {'alpha': [1.0, .1, .05, .01, .001, 0.0]}
    model = MultinomialNB()  # check multiclass config
    fit_and_test_model(data, parameters, model, results)

def logistic_regression(data, results):
    parameters = {'C': [1e4, 1e3, 1e2, 1e1, 1],
                  'penalty': ['l2','l1'],
                  'dual': [False, True]}
    model = LogisticRegression()
    fit_and_test_model(data, parameters, model, results)


def run_benchmark(data, results, benchmarks):
    print("|Tr|=%d" % data.num_devel_docs())
    print("|Te|=%d" % data.num_test_documents())
    print("|C|=%d" % data.num_categories())
    if benchmarks['linearsvm']:
        linear_svm(data, results)
    #if benchmarks['randomforest']:
    #    random_forest(data, results)
    #if benchmarks['logisticregression']:
    #    logistic_regression(data, results)
    #if benchmarks['multinomialnb']:
    #    multinomial_nb(data, results)
    #if benchmarks['knn']:
    #    knn(data, results)

if __name__ == '__main__':
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="indicates the dataset on which to run the baselines benchmark ", choices=TextCollectionLoader.valid_datasets)
    parser.add_argument("-v", "--vectordir", help="directory containing learnt vectors in .pickle format", type=str)
    parser.add_argument("-r", "--resultfile", help="path to a result container file (.csv)", type=str, default="../results.csv")
    parser.add_argument("-m", "--method", help="selects one single vectorizer method to run from "+str(TextCollectionLoader.valid_vectorizers),
                        type=str, default="all")
    parser.add_argument("--fs", help="feature selection ratio", type=float, default=0.1)
    parser.add_argument("--sublinear_tf", help="logarithmic version of the tf-like function", default=False, action="store_true")
    parser.add_argument("--global_policy", help="global policy for supervised term weighting approaches in multiclass configuration, max (default), ave, wave (weighted average), or sum", type=str, default='max')
    parser.add_argument("--classification", help="select the classification mode (binary or multiclass -- default)", type=str, default='multiclass')
    parser.add_argument("--no-linearsvm", help="removes the linearsvm classifier from the benchmark", default=False, action="store_true")
    #parser.add_argument("--no-multinomialnb", help="removes the multinomialnb classifier from the benchmark", default=False, action="store_true")
    #parser.add_argument("--no-randomforest", help="removes the randomforest classifier from the benchmark", default=False, action="store_true")
    #parser.add_argument("--no-logisticregression", help="removes the logisticregression classifier from the benchmark", default=False, action="store_true")
    #parser.add_argument("--no-knn", help="removes the knn classifier from the benchmark", default=False, action="store_true")
    args = parser.parse_args()

    err_exception(args.dataset and args.vectordir, "Specify only one run mode: runing baselines on a dataset or precalculated vectors.")

    benchmarks = dict({'linearsvm': not args.no_linearsvm,
                       #'multinomialnb': not args.no_multinomialnb,
                       #'randomforest': not args.no_randomforest,
                       #'logisticregression': not args.no_logisticregression,
                       #'knn': not args.no_knn
                       })

    if args.dataset:
        print("Runing classification benchmark on baselines\n"+"-"*80)
        results = BaselineResultTable(args.resultfile, args.classification)
        for vectorizer in ([args.method] if args.method != 'all' else TextCollectionLoader.valid_vectorizers):
            if args.classification == 'binary':
                for cat in TextCollectionLoader.valid_catcodes[args.dataset]:
                    data = TextCollectionLoader(dataset=args.dataset, vectorizer=vectorizer, rep_mode='sparse', feat_sel=args.fs,
                                                sublinear_tf=args.sublinear_tf, global_policy=args.global_policy,
                                                positive_cat=cat)
                    run_benchmark(data, results, benchmarks)
                    print('-'*80)
            elif args.classification == 'multiclass':
                data = TextCollectionLoader(dataset=args.dataset, vectorizer=vectorizer, rep_mode='sparse',
                                            feat_sel=args.fs, sublinear_tf=args.sublinear_tf, global_policy=args.global_policy)
                run_benchmark(data, results, benchmarks)
            else:
                raise ValueError('classification param should be either "multiclass" or "binary"')

    elif args.vectordir:
        print("Runing classification benchmark on learnt vectors in " + args.vectordir)
        vectors = [pickle for pickle in os.listdir(args.vectordir) if pickle.endswith('.pickle')]
        results = Learning2Weight_ResultTable(args.resultfile)
        for i,vecname in enumerate(vectors):
            print("Vector file: " + vecname)
            data = WeightedVectors.unpickle(indir=args.vectordir, infile_name=vecname)
            run_benchmark(data, results, benchmarks)
            print("Completed %d/%d" % (i,len(vectors)))
            print('-'*80)

    print("Done.")
#some results reuters21578
#tfgr, sublinear max: MacroF1=0.609 microF1=0.850
#tfgr, sublinear ave: MacroF1=0.568 microF1=0.850
#tfgr,           ave: MacroF1=0.601 microF1=0.844
