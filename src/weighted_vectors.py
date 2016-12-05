import cPickle as pickle
import numpy as np


class WeightedVectors:
    def __init__(self, method_name, from_dataset, from_category, trX, trY, vaX, vaY, teX, teY):
        self.method_name = method_name
        self.from_dataset = from_dataset
        self.from_category = from_category
        self.trX = trX
        self.trY = trY
        self.vaX = vaX
        self.vaY = vaY
        self.teX = teX
        self.teY = teY

    def pickle(self, outfile_name):
        pickle.dump(self, open(outfile_name, 'wb'))

    @staticmethod
    def unpickle(infile_name):
        return pickle.load(open(infile_name, 'rb'))

