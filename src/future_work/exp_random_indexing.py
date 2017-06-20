from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.multiclass import OneVsRestClassifier
from utils.metrics import *
from data.dataset_loader import TextCollectionLoader
from future_work.random_indexing import RandomIndexing
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
import time


def fit_model_hyperparameters(data, parameters, model):
    single_class = data.num_categories() == 1
    if not single_class:
        parameters = {'estimator__' + key: parameters[key] for key in parameters.keys()}
        model = OneVsRestClassifier(model, n_jobs=-1)
    model_tunning = GridSearchCV(model, param_grid=parameters,
                                 scoring=make_scorer(macroF1), error_score=0, refit=True, cv=3, n_jobs=-1)

    Xtr, ytr = data.get_devel_set()
    Xtr.sort_indices()
    if single_class:
        ytr = np.squeeze(ytr)

    tunned = model_tunning.fit(Xtr, ytr)

    return tunned

def fit_and_test_model(data, parameters, model):
    init_time = time.time()
    tunned_model = fit_model_hyperparameters(data, parameters, model)
    tunning_time = time.time() - init_time
    print("%s: best parameters %s, best score %.3f, took %.3f seconds" %
          (type(model).__name__, tunned_model.best_params_, tunned_model.best_score_, tunning_time))

    Xte, yte = data.get_test_set()
    #Xte.sort_indices()
    yte_ = tunned_model.predict(Xte)

    macro_f1 = macroF1(yte, yte_)
    micro_f1 = microF1(yte, yte_)
    return macro_f1, micro_f1

#print "classification / test"
def linear_svm(data, learner):
    parameters = {'C': [1e2, 1e1, 1],
                  'loss': ['squared_hinge'],
                  'dual': [True, False]}


def learner_without_gridsearch(Xtr,ytr,Xte,yte, learner, cat=-1):
    if cat!=-1:
        model = learner
        ytr = ytr[:,cat]
        yte = yte[:, cat]
    else:
        model = OneVsRestClassifier(learner, n_jobs=7)

    t_ini = time.time()
    trained_model = model.fit(Xtr, ytr)
    t_train = time.time()
    yte_ = trained_model.predict(Xte)
    t_test = time.time()
    return macroF1(yte, yte_), microF1(yte, yte_), t_train-t_ini, t_test-t_train

def sort_categories_by_prevalence(data):
    ytr, yte = data.devel.target, data.test.target
    nC = ytr.shape[1]
    ytr_prev = [np.sum(ytr[:,i], axis=0) for i in range(nC)]
    ordered = np.argsort(ytr_prev)
    data.devel.target = data.devel.target[:,ordered]
    data.test.target = data.test.target[:, ordered]



#-------------------------------------------------------------------------------------------------------------------

def q_distance(v,w,M):
    dif = v-w
    if isinstance(v, csr_matrix):
        return np.sqrt(dif.dot(M).dot(dif.T))
    else:
        return np.sqrt(np.dot(np.dot(dif,M),dif.T))

def euclidean_distance(v,w):
    dif = v - w
    if isinstance(v,csr_matrix):
        return np.sqrt(dif.dot(dif.T))[0,0]
    else:
        return np.sqrt(np.dot(dif, dif.T))

def svc_ri_kernel(random_indexer):
    P = random_indexer.projection_matrix
    R = np.absolute(P.transpose().dot(P))
    R.eliminate_zeros()
    # if R.nnz*1.0/(R.shape[0]*R.shape[1]) > 0.1:
    #     R = R.toarray()
    #     print('toarray')
    #normalize(R, axis=1, norm='max', copy=False)
    ri_kernel = lambda X, Y: (X.dot(R)).dot(Y.T)
    return SVC(kernel=ri_kernel), R

def plot_distances(X_train, XP_train, nR):
    print("pair-wise distances in original space")
    orig_dists = pairwise_distances(X_train, n_jobs=-1, metric=euclidean_distance).ravel()
    distinct = orig_dists != 0
    orig_dists = orig_dists[distinct]  # select only non-identical samples pairs
    print("pair-wise distances in ri space")
    ri_dists = pairwise_distances(XP_train, n_jobs=-1, metric=euclidean_distance).ravel()[distinct]
    print("pair-wise distances in k-ri space")
    kri_dists = pairwise_distances(XP_train, n_jobs=-1, metric=lambda x, y: q_distance(x, y, R)).ravel()[distinct]
    print("plot")
    rates_o_ri = ri_dists / orig_dists
    rates_o_kri = kri_dists / orig_dists
    print("Mean distances (RI) rate: %0.2f (%0.2f)" % (np.mean(rates_o_ri), np.std(rates_o_ri)))
    print("Mean distances (KRI) rate: %0.2f (%0.2f)" % (np.mean(rates_o_kri), np.std(rates_o_ri)))

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.subplot(121)
    plt.hexbin(orig_dists, ri_dists, gridsize=100, cmap=plt.cm.PuBu)
    plt.xlabel("Pairwise squared distances in original space")
    plt.ylabel("Pairwise squared distances in projected space")
    plt.title("Pairwise distances distribution for n_components=%d" % nR)
    plt.subplot(122)
    plt.hexbin(orig_dists, kri_dists, gridsize=100, cmap=plt.cm.PuBu)
    plt.xlabel("Pairwise squared distances in original space")
    plt.ylabel("Pairwise squared distances in projected space")
    plt.title("Pairwise distances distribution for n_components=%d" % nR)
    plt.show()


def experiment(Xtr,ytr,Xte,yte,learner,method,dataset,nF,fo,run=0):
    macro_f1, micro_f1, t_train, t_test = learner_without_gridsearch(Xtr,ytr,Xte,yte, learner)
    fo.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(method, dataset, nF, run, t_train, t_test, macro_f1, micro_f1))
    fo.flush()

with open('Kernel_RI_results.txt', 'w') as fo:
    fo.write("Method\tdataset\tnF\trun\ttrainTime\ttestTime\tMacroF1\tmicroF1\n")

    compute_distances=False
    compute_classification=True
    bow_computed = True #False
    for non_zeros in [10]:#[2, 5, 10]:
        for dataset in ['reuters21578','20newsgroups']:#TextCollectionLoader.valid_datasets:
            data = TextCollectionLoader(dataset=dataset)
            nF = data.num_features()

            if not bow_computed:
                X_train, y_train = data.get_devel_set()
                X_test,  y_test = data.get_test_set()
                experiment(X_train, y_train, X_test,  y_test, LinearSVC(), 'BoW', dataset, nF, fo)

            for run in range(5):
                for nR in [5000, 10000]:#, nF/2, nF, int(nF*1.25)]:
                    print('Running {} nR={} non-zero={}...'.format(dataset,nR,non_zeros))
                    data = TextCollectionLoader(dataset=dataset)
                    X_train, y_train = data.get_devel_set()
                    X_test, y_test = data.get_test_set()

                    print("projection")
                    random_indexing = RandomIndexing(latent_dimensions=nR, non_zeros=non_zeros, positive=False, postnorm=True)
                    XP_train = random_indexing.fit_transform(X_train)
                    XP_test  = random_indexing.transform(X_test)

                    ri_kernel, R = svc_ri_kernel(random_indexing)
                    if compute_distances:
                        plot_distances(X_train, XP_train, nR)
                    if compute_classification:
                        experiment(XP_train, y_train, XP_test, y_test, LinearSVC(), 'RI(%d)' % non_zeros, dataset, nR, fo, run=run)
                        experiment(XP_train, y_train, XP_test, y_test, ri_kernel, 'KernelRI(%d)' % non_zeros, dataset, nR, fo, run=run)
        bow_computed = True
        print('Done!')
