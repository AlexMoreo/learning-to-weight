from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.multiclass import OneVsRestClassifier
from utils.metrics import *
from data.dataset_loader import TextCollectionLoader
from future_work.random_indexing import RandomIndexing
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
import time, math, os, sys
from numpy.linalg import cholesky
from scipy.sparse import csr_matrix, csc_matrix
import pandas as pd
import cPickle as pickle
from helpers import create_if_not_exists
from sklearn.preprocessing import normalize
import scipy
import matplotlib.pyplot as plt

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

def get_cov_matrix(P):
    R = np.absolute(P.transpose().dot(P))
    R.eliminate_zeros()
    return R

def get_diagonal(P, square_root=False):
    #the diagonal of the R matrix is the L2-norm of each column vector in P
    diag = scipy.sparse.linalg.norm(P, axis=0)
    if square_root:
        diag = np.sqrt(diag)
    diag_idx = range(P.shape[1])
    return csc_matrix((diag,(diag_idx,diag_idx)))

def get_cholesky(R):
    try:
        L = csc_matrix(cholesky(R.toarray()))
        return L
    except np.linalg.linalg.LinAlgError:
        print('error: matrix is not semidefinite positive!!!')
        return None

def svc_ri_kernel(R):
    ri_kernel = lambda X, Y: (X.dot(R)).dot(Y.T)
    return SVC(kernel=ri_kernel)

def extract_diagonal(R):
    return csc_matrix(np.diag(np.diag(R.toarray())))

# if the experiment has not yet been run, then runs it
def experiment(Xtr, ytr, Xte, yte, learner, method, k, dataset, nF, df, run=0, addtime_tr=0, addtime_te=0):
    learner_name = type(learner).__name__
    if df.already_calculated(method, k, learner_name, dataset, nF, run): return
    if isinstance(Xtr, csr_matrix):
        Xtr.sort_indices()
    if isinstance(Xte, csr_matrix):
        Xte.sort_indices()
    macro_f1, micro_f1, t_train, t_test = learner_without_gridsearch(Xtr,ytr,Xte,yte, learner)
    df.add_row(method, k, learner_name, dataset, nF, run, t_train+addtime_tr, t_test+addtime_te, macro_f1, micro_f1)


def experiment_kernel_linsvc(P, X_train, y_train, X_test, y_test, method, k, dataset, nF, df, run=0, addtime_tr=0, addtime_te=0):
    if df.already_calculated(method, k, 'LinearSVC', dataset, nF, run): return

    t_ini = time.time()
    R = get_cov_matrix(P)
    L = get_cholesky(R)
    if L is not None:
        XL_train = X_train.dot(L)
        addtime_tr += (time.time() - t_ini)

        t_ini = time.time()
        XL_test = X_test.dot(L)
        addtime_te += (time.time() - t_ini)

        experiment(XL_train, y_train, XL_test, y_test, LinearSVC(), method, k, dataset, nF, results, run=run, addtime_tr=addtime_tr, addtime_te=addtime_te)
    else:
        df.add_row(method, k, 'LinearSVC', dataset, nF, run, np.NaN, np.NaN, np.NaN, np.NaN)

def experiment_kernel_svc(P, X_train, y_train, X_test, y_test, method, k, dataset, nF, df, run=0, addtime_tr=0, addtime_te=0):
    if df.already_calculated(method, k, 'SVC', dataset, nF, run): return

    t_ini = time.time()
    R = get_cov_matrix(P)
    addtime_tr += (time.time() - t_ini)
    experiment(X_train, y_train, X_test, y_test, svc_ri_kernel(R), method, k, dataset, nF, results, run=run, addtime_tr=addtime_tr, addtime_te=addtime_te)


def experiment_diag_linsvc(P, X_train, y_train, X_test, y_test, method, k, dataset, nF, df, run=0, addtime_tr=0, addtime_te=0):
    if df.already_calculated(method, k, 'LinearSVC', dataset, nF, run): return

    t_ini = time.time()
    D = get_diagonal(P, square_root=True)
    XD_train = X_train.dot(D)
    addtime_tr += (time.time() - t_ini)

    t_ini = time.time()
    XD_test = X_test.dot(D)
    addtime_te += (time.time() - t_ini)

    experiment(XD_train, y_train, XD_test, y_test, LinearSVC(), method, k, dataset, nF, results, run=run, addtime_tr=addtime_tr, addtime_te=addtime_te)

def experiment_diag_svc(P, X_train, y_train, X_test, y_test, method, k, dataset, nF, df, run=0, addtime_tr=0, addtime_te=0):
    if df.already_calculated(method, k, 'SVC', dataset, nF, run): return

    t_ini = time.time()
    D = get_diagonal(P, square_root=False)
    addtime_tr += (time.time() - t_ini)

    experiment(X_train, y_train, X_test, y_test, svc_ri_kernel(D), method, k, dataset, nF, results, run=run, addtime_tr=addtime_tr, addtime_te=addtime_te)


class KernelResults:
    def __init__(self, file, autoflush=True, verbose=False):
        self.file = file
        self.columns = ['Method', 'k', 'learner', 'dataset', 'nF', 'run', 'trainTime', 'testTime', 'MacroF1', 'microF1']
        self.autoflush = autoflush
        self.verbose = verbose
        if os.path.exists(file):
            self.tell('Loading existing file from {}'.format(file))
            self.df = pd.read_csv(file, sep='\t')
        else:
            self.tell('File {} does not exist. Creating new frame.'.format(file))
            self.df = pd.DataFrame(columns=self.columns)

    def already_calculated(self, method, k, learner, dataset, nF, run):
        return ((self.df['Method'] == method) &
                (self.df['k'] == k) &
                (self.df['learner'] == learner) &
                (self.df['dataset'] == dataset) &
                (self.df['nF'] == nF) &
                (self.df['run'] == run)).any()

    def add_row(self, method, k, learner, dataset, nF, run, trainTime, testTime, MacroF1, microF1):
        s = pd.Series([method, k, learner, dataset, nF, run, trainTime, testTime, MacroF1, microF1], index=self.columns)
        self.df = self.df.append(s, ignore_index=True)
        if self.autoflush: self.flush()
        self.tell(s.to_string())

    def flush(self):
        self.df.to_csv(self.file, index=False, sep='\t')

    def tell(self, msg):
        if self.verbose: print(msg)

    def compare_plot(self, baseline, method1, method2, with_learner, with_dataset, with_k, save_to_dir='', min_nf=2500, max_nf=15000, eval = 'MacroF1'):
        def select(method, nf=0, run=0, k=1, eval='MacroF1'):
            df=self.df
            s = df[(df['Method'] == method) & (df['learner'] == with_learner) & (df['dataset'] == with_dataset) & (df['k'] == k) & (df['run'] == run)]
            if nf > 0:
                s = s[df['nF'] == nf]
            return s.iloc[0][eval]

        nf_values = [x for x in self.df['nF'].unique() if x >= min_nf and x <= max_nf]
        run_values = self.df['run'].unique()

        bow_baseline = select(baseline, eval=eval)
        serie_1 = []
        serie_2 = []
        p_vals  = []

        for nf in nf_values:
            runs_1 = [select(method1, nf, r, with_k, eval) for r in run_values]
            runs_2 = [select(method2, nf, r, with_k, eval) for r in run_values]
            mu_1, std_1 = np.mean(runs_1), np.std(runs_1)
            mu_2, std_2 = np.mean(runs_2), np.std(runs_2)
            serie_1.append((mu_1,std_1))
            serie_2.append((mu_2, std_2))
            ts,pvalue=scipy.stats.ttest_rel(runs_1, runs_2)
            p_vals.append(pvalue)

        linewidth = 3
        fig, (axmain,axttest) = plt.subplots(2, figsize=(5.625,10))
        x_lims=[nf_values[0] - 100, nf_values[-1] + 100]

        axmain.grid(True)
        axmain.set_title(with_dataset.title())
        axmain.set_ylabel(eval)
        axmain.set_xlim(x_lims)
        axmain.errorbar(nf_values, zip(*serie_1)[0], yerr=zip(*serie_1)[1], fmt='y-o', ecolor='k', label=method1, linewidth=linewidth, elinewidth=1)
        axmain.errorbar(nf_values, zip(*serie_2)[0], yerr=zip(*serie_2)[1], fmt='r-o', ecolor='k', label=method2, linewidth=linewidth, elinewidth=1)
        axmain.plot(x_lims, [bow_baseline] * 2, 'b--', label=baseline, linewidth=linewidth)
        axmain.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False, shadow=False, ncol=5)

        axttest.grid(True)
        axttest.semilogy(nf_values, p_vals, 'k-o', linewidth=linewidth, label=method1+" vs "+method2)
        axttest.semilogy(x_lims, [0.005] * 2, 'g--', label='0.005', linewidth=linewidth)
        axttest.semilogy(x_lims, [0.001] * 2, 'g--', label='0.001', linewidth=linewidth)
        axttest.set_xlim(x_lims)
        axttest.text(x_lims[0]+200, 0.005, r'$\alpha = 0.005$', fontsize=15)
        axttest.text(x_lims[0]+200, 0.001, r'$\alpha = 0.001$', fontsize=15)
        axttest.set_xlabel('nR')
        axttest.set_ylabel('p-value')

        if save_to_dir:
            if not os.path.exists(save_to_dir):
                os.makedirs(save_to_dir)
            plot_name = method1+'_vs_'+method2 + '_' + with_learner + '_' + with_dataset + '_' + str(with_k) + '_' + eval + ".pdf"
            plt.savefig(os.path.join(save_to_dir, plot_name), format='pdf')
        else:
            plt.show()


# Checks if the random index matrix for this run has been calculated and, if so, returns it.
# If otherwise, creates a new matrix, stores it in path, and returns it.
# This method guarantees all experiments for the same run operates on the same random index matrix.
def get_random_matrix(base_path, dataset, nR, k, run, X_train, verbose=False):
    matrix_dir = os.path.join(base_path, dataset, 'nr' + str(nR), 'k' + str(k))
    matrix_path = os.path.join(matrix_dir, 'run'+str(run)+'.pickle')
    if os.path.exists(matrix_path):
        if verbose: print('Loading random matrix from {}'.format(matrix_path))
        random_matrix = pickle.load(open(matrix_path, 'rb'))
    else:
        if verbose: print('Creating random matrix and storing it in {}'.format(matrix_path))
        os.makedirs(matrix_dir)
        random_indexing = RandomIndexing(latent_dimensions=nR, non_zeros=non_zeros, positive=False, postnorm=True)
        random_indexing.fit(X_train)
        random_matrix = random_indexing.projection_matrix
        pickle.dump(random_matrix, open(matrix_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return random_matrix

r = KernelResults('../../results/Kernel_experiments.csv')
for dataset in ['ohsumed20k']:#['reuters21578', '20newsgroups', 'ohsumed20k']:
    nF = 28828 if dataset == 'reuters21578' else (26747 if dataset=='20newsgroups' else 24768)
    if dataset == 'ohsumed20k': max_nf = 10000
    for eval in ['MacroF1', 'microF1']:
        r.compare_plot('BoW', 'RI', 'RI_R', 'LinearSVC', dataset, 2.0, save_to_dir='../plots/'+dataset+'/k2', eval=eval, max_nf=max_nf)
        r.compare_plot('BoW', 'RI', 'RI_D', 'LinearSVC', dataset, 2.0, save_to_dir='../plots/'+dataset+'/k2', eval=eval, max_nf=max_nf)
        r.compare_plot('BoW', 'RI_D', 'RI_R', 'LinearSVC', dataset, 2.0, save_to_dir='../plots/'+dataset+'/k2', eval=eval, max_nf=max_nf)
        # r.compare_plot('BoW', 'RI', 'RI_R', 'LinearSVC', dataset, nF/100, save_to_dir='../plots/' + dataset + '/k1p', eval=eval)
        # r.compare_plot('BoW', 'RI', 'RI_D', 'LinearSVC', dataset, nF/100, save_to_dir='../plots/' + dataset + '/k1p', eval=eval)
        # r.compare_plot('BoW', 'RI_D', 'RI_R', 'LinearSVC', dataset, nF/100, save_to_dir='../plots/' + dataset + '/k1p',eval=eval)
#r.compare_plot('BoW','RI','RI_R','LinearSVC','reuters21578', 2.0, eval='MacroF1')
sys.exit()

matrix_dir = "../../matrices"
out_dir = "../../results"
create_if_not_exists(matrix_dir)
create_if_not_exists(out_dir)

out_file = os.path.join(out_dir, "Kernel_experiments.csv")

results = KernelResults(out_file, verbose=True)

bow_enabled = True
ri_enabled  = True

linearsvc_enabled = True
svc_enabled = False

raw_enabled = True
kernel_enabled = True
diagonal_enabled = True

for dataset in ['reuters21578','20newsgroups','ohsumed']:
    print("Dataset: {}".format(dataset))

    data = TextCollectionLoader(dataset=dataset)
    nF = data.num_features()
    X_train, y_train = data.get_devel_set()
    X_test, y_test = data.get_test_set()

    if bow_enabled:
        # here we use the co-occurrence space as the projection from which the correlations will be calculated
        P = X_train

        if linearsvc_enabled:
            if raw_enabled:
                experiment(X_train, y_train, X_test, y_test, LinearSVC(), 'BoW', 1, dataset, nF, results)
            if kernel_enabled:
                experiment_kernel_linsvc(P, X_train, y_train, X_test, y_test, 'BoW_R', 1, dataset, nF, results)
            if diagonal_enabled:
                experiment_diag_linsvc(P, X_train, y_train, X_test, y_test, 'Bow_D', 1, dataset, nF, results)
        if svc_enabled:
            if raw_enabled:
                experiment(X_train, y_train, X_test, y_test, SVC(kernel='linear'), 'BoW', 1, dataset, nF, results)
            if kernel_enabled:
                experiment_kernel_svc(P, X_train, y_train, X_test, y_test, 'BoW_R', 1, dataset, nF, results)
            if diagonal_enabled:
                experiment_diag_linsvc(P, X_train, y_train, X_test, y_test, 'Bow_D', 1, dataset, nF, results)

    if ri_enabled:
        for non_zeros in [2, -1]:
            if non_zeros == -1:
                non_zeros = nF / 100
            for run in range(10):
                for nR in [2500, 5000, 6000, 7000, 8000, 9000, 10000, 15000]:
                    print("\trun {}: nR={} k={}".format(run,nR,non_zeros))
                    if nR > nF: continue
                    P = get_random_matrix(matrix_dir, dataset, nR, non_zeros, run, X_train, verbose=True)
                    XP_train = X_train.dot(P)
                    XP_test  = X_test.dot(P)

                    if linearsvc_enabled:
                        if raw_enabled:
                            experiment(XP_train, y_train, XP_test, y_test, LinearSVC(), 'RI', non_zeros, dataset, nR, results, run=run)
                        if kernel_enabled:
                            experiment_kernel_linsvc(P, XP_train, y_train, XP_test, y_test, 'RI_R', non_zeros, dataset, nR, results, run=run)
                        if diagonal_enabled:
                            experiment_diag_linsvc(P, XP_train, y_train, XP_test, y_test, 'RI_D', non_zeros, dataset, nR, results, run=run)
                    if svc_enabled:
                        if raw_enabled:
                            experiment(XP_train, y_train, XP_test, y_test, SVC(kernel='linear'), 'RI', non_zeros, dataset, nR, results, run=run)
                        if kernel_enabled:
                            experiment_kernel_svc(P, XP_train, y_train, XP_test, y_test, 'RI_R', non_zeros, dataset, nR, results, run=run)
                        if diagonal_enabled:
                            experiment_diag_linsvc(P, XP_train, y_train, XP_test, y_test, 'RI_D', non_zeros, dataset, nR, results, run=run)


