import os
import cPickle as pickle
import numpy as np
from helpers import create_if_not_exists


class WeightedVectors:
    def __init__(self, method_name, from_dataset, from_category, trX, trY, vaX, vaY, teX, teY, run_params_dic=None):
        self.method = method_name
        self.name = from_dataset
        self.positive_cat = from_category
        self.vectorize = 'learned'
        self.trX = trX
        self.trY = trY
        self.vaX = vaX
        self.vaY = vaY
        self.teX = teX
        self.teY = teY
        self.run_params_dic = run_params_dic

    def pickle(self, outdir, outfile_name):
        create_if_not_exists(outdir)
        pickle.dump(self, open(os.path.join(outdir,outfile_name), 'wb'))

    @staticmethod
    def unpickle(indir, infile_name):
        return pickle.load(open(os.path.join(indir,infile_name), 'rb'))

    def get_train_set(self):
        return self.trX, self.trY

    def get_validation_set(self):
        return self.vaX, self.vaY

    def get_devel_set(self):
        return np.concatenate((self.trX,self.vaX)), np.concatenate((self.trY,self.vaY))

    def get_test_set(self):
        return self.teX, self.teY

    def num_features(self):
        return self.trX.shape[1]

    def num_categories(self):
        return 2

    def num_tr_documents(self):
        return len(self.trY)

    def num_val_documents(self):
        return len(self.vaY)

    def num_test_documents(self):
        return len(self.teY)

    def get_categories(self):
        return ['negative','positive']


