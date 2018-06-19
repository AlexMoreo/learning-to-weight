from numpy import log
from sklearn.linear_model import LogisticRegression
from scipy.sparse.linalg import norm
from joblib import Parallel
from joblib import delayed
from sklearn.metrics import f1_score
from cca_operations import *

k1 = 1.2
k2 = 0.
k3 = 1000.
b = 0.75

# ----------------------------------------------------------------
# Athoms
# ----------------------------------------------------------------
#
# csr_matrix:
# ----------------------------------------------------------------
def ft01_csr(tf):
    return tf.copy()

def ft02_csr(tf):
    tf = tf.copy()
    nonzeros = (tf!=0)
    tf[nonzeros] = 1+log(tf[nonzeros])
    return tf

def ft03_csr(tf):
    tf = tf.copy()
    maxtf = tf.max(axis=1).toarray().flatten()
    rows,cols = tf.nonzero()
    tf[rows, cols] = 0.5 + ((0.5 + tf[rows, cols]) / maxtf[rows])
    return tf

def ft04_csr(tf):
    tf = tf.copy()
    avgtf = tf.sum(axis=1).getA1() /  tf.getnnz(axis=1).clip(1.) # the .mean takes into account all 0 values
    lg_avgtf = np.log(avgtf.clip(1.))
    rows,cols = tf.nonzero()
    tf[rows,cols] = ((1. + log(tf[rows, cols])) / (1. + lg_avgtf[rows]))
    return tf

def ft05_csr(tf):
    tf = tf.copy()
    dl = tf.sum(axis=1).getA1()
    avgdl = dl.mean()
    rows,cols = tf.nonzero()
    tf[rows,cols] = ((k1+1)*tf[rows,cols]) / ( (k1*((1-b)+b*dl[rows]/avgdl) + dl[rows]) + tf[rows,cols])
    return tf

# dense row vector
# ---------------------------------
def __df(tf):
    #the clip is useful to prevent div.by zero, which could happen because the tf can be a split of the training set, so
    #it is not guaranteed to have all columns non-empty
    return (tf > 0).toarray().sum(axis=0).clip(1)

def ft06_row(tf):
    N = tf.shape[0]
    idf = log(float(N) / __df(tf))
    return idf.reshape(1,-1)
    #return np.tile(idf, (N,1))

def ft07_row(tf):#, notile=False):
    N = tf.shape[0]
    idf = log(1. + float(N) / __df(tf))
    return idf.reshape(1, -1)
    # if notile:
    #     return idf
    # else:
    #     return np.tile(idf, (N, 1))

def ft08_row(tf):
    N = tf.shape[0]
    idf = log((0.5 + float(N) - __df(tf))/0.5)
    # return np.tile(idf, (N, 1))
    return idf.reshape(1, -1)

def ft09_row(tf):
    N = tf.shape[0]
    df = __df(tf)
    idf = log((0.5 + float(N) - df)/(df + 0.5))
    # return np.tile(idf, (N, 1))
    return idf.reshape(1, -1)

def ft10_row(tf):
    N = tf.shape[0]
    df = __df(tf)
    idf = log((float(N) - df)/df)
    # return np.tile(idf, (N, 1))
    return idf.reshape(1, -1)

def ft11_row(tf):
    N = tf.shape[0]
    df = __df(tf)
    idf = log((float(N) + 0.5)/df) / log(N+1.)
    # return np.tile(idf, (N, 1))
    return idf.reshape(1, -1)

# dense column vector
# --------------------------------
def ft12_col(tf):
    tffactor = ft01_csr(tf)
    idffactor = ft07_row(tf).squeeze()
    rows,cols = tf.nonzero()
    tfidf = tffactor.copy()
    tfidf[rows,cols] = np.multiply(tffactor[rows,cols], idffactor[cols])
    norms = norm(tfidf , ord=2, axis=1)
    norms[norms == 0] = 1.
    cosnorm = 1. / norms
    return cosnorm.reshape(-1, 1)

def ft13_col(tf):
    tffactor = ft02_csr(tf)
    idffactor = ft07_row(tf).squeeze()
    rows,cols = tf.nonzero()
    tfidf = tffactor.copy()
    tfidf[rows,cols] = np.multiply(tffactor[rows,cols], idffactor[cols])
    norms = norm(tfidf, ord=2, axis=1)
    norms[norms==0] = 1. #empty documents after feature selection could exist
    cosnorm = 1. / norms
    return cosnorm.reshape(-1, 1)

def ft14_col(tf): # average document length in bytes
    dl = tf.sum(axis=1)*6. # the average number of chars per word is around 6
    dl = dl.clip(min=1) # to avoid possible divisions by 0
    return dl.reshape(-1, 1)


# helpers to obtain the best slope for the pivot-length normalization (t15, t16, and t17)
# --------------------------------
def _t15_denom(slope, t13, avgt13):
    return (1. - slope) + slope * (avgt13 / t13)

def _t16_denom(slope, dl, avgdl):
    return (1. - slope) * avgdl + slope * dl

def _t17_denom(slope, n_unique_terms, pivot):
    return (1. - slope) * pivot + slope * n_unique_terms

def _score_normalization_lr(X,y,Xva,yva):
    lr = LogisticRegression(C=1000) # some normalization functions work bad with C=1
    lr.fit(X, y)
    yva_ = lr.predict(Xva)
    return f1_score(yva, yva_)

def _slope_t15_eval(slope, Xtr, ytr, t13, avgt13, Xva, yva, t13va, avgt13va):
    Xtr_normalized = division(Xtr, _t15_denom(slope, t13, avgt13))
    Xva_normalized = division(Xva, _t15_denom(slope, t13va, avgt13va))
    return _score_normalization_lr(Xtr_normalized, ytr, Xva_normalized, yva)

def _slope_t16_eval(slope, Xtr, ytr, dl, avgdl, Xva, yva, dlva):
    Xtr_normalized = division(Xtr, _t16_denom(slope, dl, avgdl))
    Xva_normalized = division(Xva, _t16_denom(slope, dlva, avgdl))
    return _score_normalization_lr(Xtr_normalized, ytr, Xva_normalized, yva)

def _slope_t17_eval(slope, Xtr, ytr, n_unique_terms, Xva, yva, n_unique_terms_va):
    pivot_tr = n_unique_terms.mean()
    Xtr_normalized = division(Xtr, _t17_denom(slope, n_unique_terms, pivot_tr))
    pivot_va = n_unique_terms_va.mean()
    Xva_normalized = division(Xva, _t17_denom(slope, n_unique_terms_va, pivot_va))
    return _score_normalization_lr(Xtr_normalized, ytr, Xva_normalized, yva)

def find_best_slope_t15(Xtr, ytr, Xva, yva, slopes=np.arange(0.1, 1.1, 0.1)): #we skip 0 since this equates to doing nothing (so returning t01(tf))
    t13 = ft13_col(Xtr)
    avgt13 = t13.mean()

    t13va = ft13_col(Xva)
    avgt13va = t13va.mean()

    scores = Parallel(n_jobs=-1, backend="threading")(
        delayed(_slope_t15_eval)(slope, Xtr, ytr, t13, avgt13, Xva, yva, t13va, avgt13va) for slope in slopes
    )
    best_slope = slopes[np.argsort(scores)[-1]]

    return best_slope

def find_best_slope_t16(Xtr, ytr, Xva, yva, slopes=np.arange(0.1, 1.1, 0.1)): #we skip 0 since this equates to doing nothing (so returning t01(tf))
    dl = ft14_col(Xtr)
    avgdl = dl.mean()

    dlva = ft14_col(Xva)

    scores = Parallel(n_jobs=-1, backend="threading")(
        delayed(_slope_t16_eval)(slope, Xtr, ytr, dl, avgdl, Xva, yva, dlva) for slope in slopes
    )
    best_slope = slopes[np.argsort(scores)[-1]]

    return best_slope

def find_best_slope_t17(Xtr, ytr, Xva, yva, slopes=np.arange(0.1, 1.1, 0.1)): #we skip 0 since this equates to doing nothing (so returning t01(tf))
    n_unique_terms = (Xtr>0).sum(axis=1)
    n_unique_terms_va = (Xva > 0).sum(axis=1)

    scores = Parallel(n_jobs=-1, backend="threading")(
        delayed(_slope_t17_eval)(slope, Xtr, ytr, n_unique_terms, Xva, yva, n_unique_terms_va) for slope in slopes
    )
    best_slope = slopes[np.argsort(scores)[-1]]

    return best_slope

def ft15_col(tf, slope_t15):
    t13 = ft13_col(tf)
    avgt13 = t13.mean()
    return 1. / _t15_denom(slope_t15, t13, avgt13)

def ft16_col(tf, slope_t16):
    dl = ft14_col(tf)
    avgdl = dl.mean()
    return 1. / _t16_denom(slope_t16, dl, avgdl)

def ft17_col(tf, slope_t16):
    n_unique_terms = (tf > 0).sum(axis=1)
    return 1. / _t17_denom(slope_t16, n_unique_terms, pivot=n_unique_terms.mean())

def ft18_csr(tf):
    dl = ft14_col(tf)
    avgdl = dl.mean()
    factor = (k1*(1.-b)+b*(dl/avgdl))
    denom = addition(tf, factor)
    return division(1.,denom)

def ft19(tf): # not applicable to text classification
    pass
    # qtf = ? not applicable to text classification
    # numer = multiply(k3+1.,qtf)
    # denom = addition(qtf, k3)
    # return division(numer, denom)

def ft20(tf): # not applicable to text classification
    pass
    # qtf = ?
    # maxqtf = ?
    # rows,cols = qtf.nonzero()
    # qtf[rows, cols] = 0.5 + ((0.5 + qtf[rows, cols]) / maxqtf[rows])
    # return qtf

# constant random float in [0., 99.]
def ct21_float(tf=None):
    def _constant():
        return np.random.rand()*100.0
    return _constant

