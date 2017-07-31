from math import sqrt, pi
from scipy.stats import norm as N
from future_work.random_indexing import RandomIndexing
import numpy as np
from numpy.linalg import cholesky
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
import sys

def analytical():
    nF = 28000
    nR = 10000
    k = 2

    nF = float(nF)
    nR = float(nR)
    k = float(k)

    mu_Rii = nF/nR
    var_Rii = nF*(nR-k)/(nR*nR*k)

    mu_dotij = 0
    var_dotij = nF*(k-1)/(k*nR*(nR-1))

    mu_Rij = sqrt(var_dotij)*sqrt(2/pi)
    var_Rij = var_dotij*(1-2/pi)

    mu_z = mu_Rii - (nR-1)*mu_Rij
    var_z = var_Rii + (nR-1)*var_Rij

    z_score = 1.0 - N(mu_z,var_z).cdf(0)

    print("nF=%d nR=%d k=%d:" % (int(nF), int(nR), int(k)))
    print("mu_z=%f" % mu_z)
    print("var_z=%f" % var_z)
    print("z-score=%.20f" % z_score)

def empirical(nF, nR, k, max_trials=100):

    def is_positive_definite(x):
        return np.all(np.linalg.eigvals(x.todense()) > 0)

    def is_positive_definite_cholesky(x):
        try:
            _ = cholesky(x.todense())
            return True
        except np.linalg.linalg.LinAlgError:
            return False

    success = 0
    trials  = 0

    X = np.eye(nF)

    for _ in range(max_trials):
        ri = RandomIndexing(latent_dimensions=nR, non_zeros=k)
        ri.fit(X)
        P = ri.projection_matrix
        R = np.absolute((P.transpose()).dot(P))
        success += 1.0 if is_positive_definite_cholesky(R) else 0.0
        trials += 1.0
        if trials % (max_trials/20) == 0:
            print('\t{}/{}'.format(trials,max_trials))

    prob = success / trials

    return prob


analytical()
sys.exit()


#p = empirical(nF=2500, nR=10, k=10, max_trials=1000)
#print("Probability = {}".format(p))

X = np.eye(4)

P = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.1, 0.9]
])
normalize(P, axis=1, copy=False, norm='l2')




R = np.absolute(P.transpose().dot(P))
X_ = X.dot(P)
L = cholesky(R)
X__ = X_.dot(L)

def n_dist(X):
    d = cdist(X,X)
    return d / np.max(d)

print('R')
print(R)

print('L')
print(L)

print('X:dist')
print(n_dist(X))

print('X_:dist')
print(X_)
print(n_dist(X_))

print('X__:dist')
print(X__)
print(n_dist(X__))



