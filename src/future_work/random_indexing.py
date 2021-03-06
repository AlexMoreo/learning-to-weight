import dill
import math
import random as rn
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize

class RandomIndexing(object):

    def __init__(self, latent_dimensions, non_zeros=2, positive=False, postnorm=False):
        self.latent_dimensions = latent_dimensions
        self.non_zeros = non_zeros
        self.positive = positive
        self.postnorm=postnorm

    #the round_dim prevents empty dimensions
    def _get_random_index_csr(self, round_dim=-1):
        non_zero_dim_value = {}
        val = 1.0 / math.sqrt(self.non_zeros)
        while len(non_zero_dim_value) < self.non_zeros:
            if round_dim != -1:
                rand_dim = round_dim
                round_dim = -1
            else:
                rand_dim = rn.randint(0, self.latent_dimensions-1)
            if rand_dim not in non_zero_dim_value:
                non_zero_dim_value[rand_dim] = +val
                if not self.positive and rn.random() < 0.5:
                    non_zero_dim_value[rand_dim] *= -1
        #rand_vector = np.zeros(self.latent_dimensions)
        #rand_vector[non_zero_dim_value.keys()] = non_zero_dim_value.values()

        return non_zero_dim_value.keys(), non_zero_dim_value.values()

    # def get_random_index_dictionary(self, coocurrence_matrix):
    #     original_dim = coocurrence_matrix.shape[1]
    #     return {i: self.get_random_index_csr(round_dim=i % self.latent_dimensions) for i in range(original_dim)}

    # def _as_projection_matrix(self, random_dic):
    #     return csc_matrix(np.vstack([rand_vec for dim, rand_vec in sorted(random_dic.items())]))

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def fit(self, X, y=None):
        if not hasattr(self, "projection_matrix"):
            nD,nF = X.shape
            values, rows, columns = [],[],[]
            for f in range(nF):
                non_zero_dim, value = self._get_random_index_csr(round_dim= f % self.latent_dimensions)
                rows += [f]*len(non_zero_dim)
                columns += non_zero_dim
                values += value
            self.projection_matrix = csc_matrix((values, (rows,columns)),shape=(nF,self.latent_dimensions))
            #self.projection_matrix = self._as_projection_matrix(self.get_random_index_dictionary(X))
        else:
            raise ValueError("Error: projection matrix is already calculated.")
        return self

    def transform(self, X, y=None):
        if not hasattr(self, "projection_matrix"):
            raise ValueError("Error: transform method called before fit.")
        projection = X.dot(self.projection_matrix)
        if self.postnorm:
            normalize(projection, axis=1, copy=False, norm='l2')
        return projection

    def count_nonzeros(self, matrix):
        if isinstance(matrix, csc_matrix) or isinstance(matrix, csr_matrix):
            return matrix.nnz
        else:
            return np.count_nonzero(matrix)

    def proj_density(self):
        if not hasattr(self, "projection_matrix"):
            raise ValueError("Error: density method called before fit.")
        return self.density(self.projection_matrix)

    def density(self, matrix):
        return self.count_nonzeros(matrix)*1.0/ np.prod(matrix.shape)


