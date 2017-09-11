import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import sys
from data.dataset_loader import TextCollectionLoader
from feature_selection.tsr_function import *
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from utils.metrics import *
import gc
from utils.result_table import AB_Results

"""
The idea is to create a classifier which leverages the supervised statiscis from the features (e.g., the tp, fp, tn, fn).
"""
def train_test_svm(Xtr, ytr, Xte, yte):
    ytr = np.squeeze(ytr)
    yte = np.squeeze(yte)
    Xtr = csr_matrix(Xtr)
    Xte = csr_matrix(Xte)
    parameters = {'C': [1e3, 1e2, 1e1, 1e0, 1e-1]}
    model = LinearSVC()
    model_tunning = GridSearchCV(model, param_grid=parameters,
                                 scoring=make_scorer(macroF1), error_score=0, refit=True, cv=5, n_jobs=-1)

    tunned = model_tunning.fit(Xtr, ytr)
    print(tunned.best_params_)
    yte_ = tunned.predict(Xte)
    cell = single_metric_statistics(np.squeeze(yte), yte_)
    fscore = f1(cell)
    return fscore, cell.tp, cell.tn, cell.fp, cell.fn

class Classifier_Results_Local(AB_Results):
    def __init__(self, file, autoflush=True, verbose=False):
        columns = ['dataset', 'category', 'method', 'run', 'f1', 'tp', 'tn', 'fp', 'fn']
        super(Classifier_Results_Local, self).__init__(file=file, columns=columns, autoflush=autoflush, verbose=verbose)

# Model
class TCClassifierNet(nn.Module):
    def __init__(self, input_size, num_classes, hidden=1000, drop_p=0.2, supervised_statistics=None):
        super(TCClassifierNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden, bias=True)
        self.linear2 = nn.Linear(hidden, hidden, bias=True)
        self.linear3 = nn.Linear(hidden, num_classes, bias=True)
        self.training = False
        self.drop_p = drop_p
        self.supervised_statistics = supervised_statistics
        if self.supervised_statistics is not None:
            info_by_feat = self.supervised_statistics.size() [1]
            self.sup_linear1 = nn.Linear(info_by_feat+1, 100)
            self.sup_linear2 = nn.Linear(100, 1)

    def forward(self, x):
        if self.supervised_statistics is not None:
            nD,nF = x.size()
            xflat = x.view(nD*nF,1)
            sfeat = self.supervised_statistics.repeat(nD, 1)
            x_feat = torch.cat([xflat,sfeat],1)
            x_h = self.sup_linear1(x_feat)
            x_h = F.relu(x_h)
            x_o = self.sup_linear2(x_h).view(nD,nF) # x should be 0 if the original input is 0 <-----------------
            x_ones = torch.gt(x,0.0).type(torch.FloatTensor).cuda()
            x = x_o * x_ones
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, self.training)
        x = self.linear2(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, self.training)
        x = self.linear3(x)
        return torch.squeeze(x)

    def train(self, X, y, optimizer, criterion, batch_size=50):
        nD = X.size()[0]
        num_batches = nD // batch_size
        loss_ = 0.0
        self.training = True
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            optimizer.zero_grad()
            outputs = model(X[start:end])
            labels = y[start:end]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_+= loss.data[0]
        self.training = False
        return loss_ / num_batches

    def test(self, X, y, criterion=None, batch_size=50):
        loss, correct, total = 0, 0, 0
        nD = X.size()[0]
        num_batches = nD // batch_size
        predictions = np.zeros((nD))
        if nD % batch_size: num_batches += 1
        for k in range(num_batches):
            start, end = k * batch_size, min((k + 1) * batch_size, nD)
            labels = y[start:end]
            outputs = self(X[start:end])
            loss += criterion(outputs, labels) if criterion else 0
            predicted = torch.round(torch.sigmoid(outputs).data)
            predictions[start:end] = predicted.cpu().numpy()
        loss /= num_batches

        if criterion is None:
            return predictions
        else:
            return predictions, loss.data[0]

    @staticmethod
    def method_name():
        return 'TorchClassifier'

class EarlyStop:
    def __init__(self, patience=5):
        self.best_loss = None
        self.patience = patience
        self.my_patience = patience

    def check(self, valid_loss):
        if self.best_loss is None:
            self.best_loss = valid_loss
        elif valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.my_patience = self.patience
        else:
            self.my_patience -= 1
        return self.my_patience == 0


def as_variables(Xy, volatile=True):
    X,y=Xy
    X = Variable(torch.from_numpy(X.astype(float)).float(), requires_grad=False, volatile=volatile).cuda()
    y = Variable(torch.from_numpy(np.squeeze(y).astype(float)).float(), requires_grad=False, volatile=volatile).cuda()
    return X,y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="indicates the dataset on which to run the baselines benchmark",
                        choices=TextCollectionLoader.valid_datasets)
    parser.add_argument("-c", "--category", help="indicates the positive category", type=int)
    parser.add_argument("-r", "--resultfile", help="path to a result container file (.csv)", type=str, default="../results/TorchClassif.results.csv")
    parser.add_argument("--fs", help="feature selection ratio (default 1.0)", type=float, default=None)
    parser.add_argument("--sublinear_tf", help="logarithmic version of the tf-like function", default=False, action="store_true")
    parser.add_argument("--force", help="forces to compute the result if already calculated", default=False, action="store_true")
    parser.add_argument("--run", help="run of the experiment", default=0, type=int)
    parser.add_argument("--feat_info", help="use supervised feature statistics", default=False, action="store_true")
    args = parser.parse_args()

    num_epochs = 1000
    hidden = 1024
    learning_rate = 0.01
    batch_size = 64
    patience = 5
    fs = args.fs
    dataset = args.dataset
    weight_baselines = ['tfidf', 'tfchi2', 'tfig', 'tf', 'binary', 'tfrf', 'l1']
    tf_mode = 'Log' if args.sublinear_tf else ''

    torch.backends.cudnn.benchmark = True
    results = Classifier_Results_Local(args.resultfile, autoflush=True, verbose=True)

    print("Running %s:%d" % (dataset, args.category))

    method_name = TCClassifierNet.method_name() + ('_STW' if args.feat_info else '')
    if args.force or not results.already_calculated(dataset=args.dataset, category=args.category, method=method_name, run=args.run):
        data = TextCollectionLoader(dataset=dataset, rep_mode='dense', vectorizer='tf', norm='none', positive_cat=args.category, feat_sel=fs, sublinear_tf=False)
        nD = data.num_devel_documents()
        m = None
        if args.feat_info:
            m = Variable(torch.from_numpy(np.array(
                [[x.tp * 1. / nD, x.fp * 1. / nD, x.fn * 1. / nD] for x in np.squeeze(data.get_4cell_matrix())],
                dtype=np.float32)), requires_grad=False, volatile=False).cuda()
        #trX, trY = as_variables(data.get_train_set(), volatile=False)
        #vaX, vaY = as_variables(data.get_validation_set())
        trX, trY = as_variables(data.get_devel_set(), volatile=False)
        teX, teY = as_variables(data.get_test_set())

        nD, nF = trX.size()
        nC = 1
        print("nD={}, nF={}".format(nD,nF))

        model = TCClassifierNet(input_size=nF, num_classes=nC, hidden=hidden, supervised_statistics=m).cuda()

        # Loss and Optimizer (Sigmoid is internally computed.)
        criterion = nn.BCEWithLogitsLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.)

        # Training the Model
        early_stop = EarlyStop(patience=patience)
        for epoch in range(num_epochs):
            trLoss = model.train(trX, trY, optimizer, criterion, batch_size)
            #vaY_, vaLoss = model.test(vaX, vaY, criterion, batch_size)
            #fscore = f1(single_metric_statistics(vaY.data.cpu().numpy(), vaY_))

            #print ('Epoch: [%d/%d], Loss: %.8f [valLoss: %.8f valF1: %.4f]' % (epoch + 1, num_epochs, trLoss, vaLoss, fscore))
            print ('Epoch: [%d/%d], Loss: %.8f' % (epoch + 1, num_epochs, trLoss))

            #if early_stop.check(vaLoss):
            if early_stop.check(trLoss):
                print("Early stop after %d steps without any improvement in the validation set" % patience)
                break
            else: #shuffle
                perm = torch.randperm(nD).cuda()
                trX = trX[perm]
                trY = trY[perm]

        # Test the Model
        teY_ = model.test(teX,teY,batch_size=batch_size)
        cell = single_metric_statistics(teY.data.cpu().numpy(), teY_)
        fscore = f1(cell)
        print('Test F1: %.3f %%' % fscore)
        results.add_row(dataset=args.dataset, category=args.category, method=method_name, run=args.run, f1=fscore, tp=cell.tp, tn=cell.tn, fp=cell.fp, fn=cell.fn)

        del trX, teX, trY, teY, model, criterion, optimizer
        gc.collect()

    # ---------------------------------------------------------------------
    for baseline in weight_baselines:
        baseline_name = tf_mode + baseline
        if not results.already_calculated(dataset=args.dataset, category=args.category, method=baseline_name):
            print('\tRunning baseline %s'%baseline)
            data = TextCollectionLoader(dataset=dataset, rep_mode='dense', vectorizer=baseline, norm='l2' if baseline!='l1' else 'none', positive_cat=args.category, feat_sel=fs,
                                        sublinear_tf=args.sublinear_tf if baseline!='l1' else False)
            trX, trY = data.get_devel_set()
            teX, teY = data.get_test_set()
            fscore, tp, tn, fp, fn = train_test_svm(trX, trY, teX, teY)
            results.add_row(dataset=args.dataset, category=args.category, method=baseline_name, run=0, f1=fscore, tp=tp, tn=tn, fp=fp, fn=fn)
