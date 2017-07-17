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
from numpy.linalg import cholesky
from scipy.sparse import csr_matrix, csc_matrix

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

#-------------------------------------------------------------------------------------------------------------------

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

def get_cov_matrix(random_indexer):
    P = random_indexer.projection_matrix
    R = np.absolute(P.transpose().dot(P))
    R.eliminate_zeros()
    return R

def get_cholesky(R):
    try:
        L = csc_matrix(cholesky(R.toarray()))
        return L
    except np.linalg.linalg.LinAlgError:
        print('error: matrix is not semidefinite positive!!!')
        raise

def svc_ri_kernel(R):
    ri_kernel = lambda X, Y: (X.dot(R)).dot(Y.T)
    return SVC(kernel=ri_kernel)

def extract_diagonal(R):
    return csc_matrix(np.diag(np.diag(R.toarray())))

def experiment(Xtr,ytr,Xte,yte,learner,method,k,dataset,nF,fo,run=0, addtime_tr=0, addtime_te=0):
    macro_f1, micro_f1, t_train, t_test = learner_without_gridsearch(Xtr,ytr,Xte,yte, learner)
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(method,k, dataset, nF, run, t_train+addtime_tr, t_test+addtime_te, macro_f1, micro_f1))
    fo.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(method,k, dataset, nF, run, t_train+addtime_tr, t_test+addtime_te, macro_f1, micro_f1))
    fo.flush()

with open('Kernel_RI_results.txt', 'w') as fo:
    fo.write("Method\tk\tdataset\tnF\trun\ttrainTime\ttestTime\tMacroF1\tmicroF1\n")

    compute_distances=False
    compute_classification=True
    bow_computed = False
    errors=0
    for dataset in ['reuters21578','20newsgroups','ohsumed20k']:#TextCollectionLoader.valid_datasets:
        for non_zeros in [2,-1]:  # [2, 5, 10]:
            data = TextCollectionLoader(dataset=dataset)
            nF = data.num_features()
            if non_zeros == -1:
                non_zeros = nF/100

            if not bow_computed:
                X_train, y_train = data.get_devel_set()
                X_test,  y_test = data.get_test_set()
                experiment(X_train, y_train, X_test,  y_test, LinearSVC(), 'BoW', 1, dataset, nF, fo)

            for run in range(5):
                for nR in [2500,5000,6000,7000,8000,9000,10000,15000]:
                    if nR > nF: continue
                    print('Running {} nR={} non-zero={}...'.format(dataset,nR,non_zeros))
                    data = TextCollectionLoader(dataset=dataset)
                    X_train, y_train = data.get_devel_set()
                    X_test, y_test = data.get_test_set()

                    print("random projection...")
                    spd = False
                    patience = 10
                    while not spd and patience > 0:
                        random_indexing = RandomIndexing(latent_dimensions=nR, non_zeros=non_zeros, positive=False, postnorm=True)
                        random_indexing.fit(X_train)
                        R = get_cov_matrix(random_indexing)
                        spd = True


                        # ri_kernel = svc_ri_kernel(random_indexing)
                        # print('Cholesky')
                        # try:
                        #     t_ini = time.time()
                        #     L = get_cholesky(R)
                        #     t_cholesky_tr = time.time()-t_ini
                        #     print('done')
                        #     spd = True
                        # except np.linalg.linalg.LinAlgError:
                        #     print('error: matrix is not semidefinite positive!!!')
                        #     errors += 1
                        #     patience -= 1

                    XP_train = random_indexing.transform(X_train)
                    XP_test  = random_indexing.transform(X_test)
                    
                    if compute_distances:
                        plot_distances(X_train, XP_train, nR)
                    if compute_classification:
                        #experiment(XP_train, y_train, XP_test, y_test, LinearSVC(), 'RI', non_zeros, dataset, nR, fo, run=run)

                        Ld = get_cholesky(extract_diagonal(R))
                        XPL_train =XP_train.dot(Ld)
                        XPL_test = XP_test.dot(Ld)
                        experiment(XPL_train, y_train, XPL_test, y_test, LinearSVC(), 'KRId', non_zeros, dataset, nR, fo, run=run)

                        # if spd:
                        #     t_ini = time.time()
                        #     XPL_train = XP_train.dot(L)
                        #     t_cholesky_tr += (time.time() - t_ini)
                        #     t_ini = time.time()
                        #     XPL_test = XP_test.dot(L)
                        #     t_cholesky_te = (time.time() - t_ini)
                        #     print("[done]")
                        #     experiment(XPL_train, y_train, XPL_test, y_test, LinearSVC(), 'KRIch', non_zeros, dataset, nR, fo, run=run, addtime_tr = t_cholesky_tr, addtime_te=t_cholesky_te)
                        #else:
                        #    experiment(XP_train, y_train, XP_test, y_test, ri_kernel, 'KRI', non_zeros, dataset, nR, fo, run=run)
                        print('end training')
        bow_computed = True
        print('Done! (with {} errors)'.format(errors))
