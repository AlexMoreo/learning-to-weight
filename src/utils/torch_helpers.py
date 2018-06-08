import torch
import numpy as np
from torch.autograd import Variable

class EarlyStop:
    def __init__(self, patience=5):
        self.best_loss = None
        self.patience = patience
        self.my_patience = patience
        self.has_improved = False

    def __compare(self, perf_measure, previous, low_is_better):
        if perf_measure == previous: return 0
        if low_is_better:
            return 1 if perf_measure < previous else -1
        else:
            return 1 if perf_measure > previous else -1

    # allows to check for more than one measures, e.g.: perf_measures[.5, .75] and low_is_better=[True, False] with
    # precedent best_loss = [.5, .6] updates the internal state and restores patience
    def check(self, perf_measures, low_is_better=True):
        if self.best_loss is None:
            self.best_loss = perf_measures
            self.has_improved = True
        elif isinstance(perf_measures, list):
            for i in range(len(perf_measures)):
                cmp = self.__compare(perf_measures[i], self.best_loss[i], low_is_better[i])
                if cmp == 1:
                    self.best_loss[i] = perf_measures[i]
                    self.best_loss[i+1:] = [perf_measures[j] if self.__compare(perf_measures[j], self.best_loss[j], low_is_better[j]) == 1 else self.best_loss[j] for j in range(i+1, len(perf_measures))]
                    self.my_patience = self.patience
                    self.has_improved = True
                    break
                elif cmp == 0: continue
                elif cmp == -1: break
            if cmp <= 0:
                self.my_patience -= 1
                self.has_improved = False
        else:
            return self.check([perf_measures],[low_is_better])
        return self.my_patience == 0

def as_variables(Xy, volatile=True):
    X,y=Xy
    X = Variable(torch.from_numpy(X.astype(float)).float(), requires_grad=False, volatile=volatile).cuda()
    y = Variable(torch.from_numpy(np.squeeze(y).astype(float)).float(), requires_grad=False, volatile=volatile).cuda()
    return X,y