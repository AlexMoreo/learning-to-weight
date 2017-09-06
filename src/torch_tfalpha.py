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
The idea is to learn an alpha parameter for each feature. Each feature will be represented as the tf (or LogTf), and the
weight is simply computed as tf * alpha, where there is a different alpha for feature. On top, there is a logistic
regressor which attempts to better separate positive and negative examples. Provided that this alpha is the optimal
idf-like score for that trainset, one could study whether there is a real correlation among the alpha distribution and
any TSR (acting as idf-like score) distribution.
Notes:
- This might allow to quantify the importance of each feature: could be used as feature selection
- The alpha-vector shall be normalized after each step, in order not to delegate the feature importance on the logistic
regressor matrix.
- If the alpha-vector demonstrates to be a better idf-like weight, then one could try to learn from the (tpr,fpr)
statistics a function that computes the alpha, and observe the function plot.
"""

def get_tsr_statistics(data, tsr_function):
    nC = data.num_categories()
    matrix_4cell = data.get_4cell_matrix()
    tsr = get_tsr_matrix(matrix_4cell, tsr_function)
    if nC == 1:
        tsr = tsr.squeeze()
    return tsr

def train_test_svm(Xtr, ytr, Xte, yte):
    ytr = np.squeeze(ytr)
    yte = np.squeeze(yte)
    Xtr = csr_matrix(Xtr)
    Xte = csr_matrix(Xte)
    parameters = {'C': [1e3,1e2, 1e1, 1, 1e-1,1e-2]}
    model = LinearSVC()
    model_tunning = GridSearchCV(model, param_grid=parameters,
                                 scoring=make_scorer(macroF1), error_score=0, refit=True, cv=5, n_jobs=-1)
    #print(model_tunning.best_estimator_.get_params())
    tunned = model_tunning.fit(Xtr, ytr)
    yte_ = tunned.predict(Xte)
    cell = single_metric_statistics(np.squeeze(yte), yte_)
    fscore = f1(cell)
    return fscore, cell.tp, cell.tn, cell.fp, cell.fn

class Alpha_Results_Local(AB_Results):
    def __init__(self, file, autoflush=True, verbose=False):
        columns = ['dataset', 'category', 'method', 'run', 'f1', 'tp', 'tn', 'fp', 'fn']
        super(Alpha_Results_Local, self).__init__(file=file, columns=columns, autoflush=autoflush, verbose=verbose)

# Model
class AlphaNet(nn.Module):
    def __init__(self, input_size, num_classes, hidden=100):
        super(AlphaNet, self).__init__()
        #self.alpha = Parameter(torch.Tensor([1.0] * nF))
        self.linear1 = nn.Linear(input_size, num_classes, bias=True)
        # self.linear2 = nn.Linear(hidden, num_classes, bias=True)
        # self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        # xalpha = x * self.alpha
        # xalpha_n = F.normalize(xalpha,p=2,dim=1)
        # h = self.linear1(xalpha_n)
        # h = self.relu(h)
        # out = self.linear2(h)
        #
        # xk = torch.mm(x * self.alpha, self.Xtr*self.alpha)
        # o = self.linear1(xk)
        return torch.squeeze(out)


    def train(self, X, y, optimizer, criterion, batch_size=50):
        nD = X.size()[0]
        num_batches = nD // batch_size
        loss_ = 0.0
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            optimizer.zero_grad()
            outputs = model(X[start:end])
            labels = y[start:end]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_+= loss.data[0]
        return loss_ / num_batches

    def test(self, X, y, criterion=None, batch_size=50):
        loss, correct, total = 0, 0, 0
        nD = X.size()[0]
        num_batches = nD // batch_size
        if nD % batch_size: num_batches += 1
        for k in range(num_batches):
            start, end = k * batch_size, min((k + 1) * batch_size, nD)
            labels = y[start:end]
            outputs = self(X[start:end])
            loss += criterion(outputs, labels) if criterion else 0
            predicted = torch.round(torch.sigmoid(outputs).data)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        acc = 100.0 * correct / total
        loss /= num_batches

        if criterion is None:
            return acc
        else:
            return acc, loss.data[0]

    @staticmethod
    def method_name():
        return 'AlphaOptimizer'

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


def as_variables(Xy):
    X,y=Xy
    X = Variable(torch.from_numpy(X.astype(float)).float(), requires_grad=False).cuda()
    y = Variable(torch.from_numpy(np.squeeze(y).astype(float)).float(), requires_grad=False).cuda()
    return X,y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="indicates the dataset on which to run the baselines benchmark",
                        choices=TextCollectionLoader.valid_datasets)
    parser.add_argument("-c", "--category", help="indicates the positive category", type=int)
    parser.add_argument("-r", "--resultfile", help="path to a result container file (.csv)", type=str, default="../results/TorchAlpha.results.csv")
    parser.add_argument("--fs", help="feature selection ratio (default 1.0)", type=float, default=None)
    parser.add_argument("--sublinear_tf", help="logarithmic version of the tf-like function", default=False, action="store_true")
    parser.add_argument("--run", help="run of the experiment", default=0, type=int)
    args = parser.parse_args()

    num_epochs = 1000
    hidden = 100
    learning_rate = 0.1
    batch_size = 50
    patience = 5
    fs = args.fs
    dataset = args.dataset
    weight_baselines = ['tfidf', 'tfchi2', 'tfig', 'tf', 'binary', 'tfrf']
    tf_mode = 'Log' if args.sublinear_tf else ''

    torch.backends.cudnn.benchmark = True
    results = Alpha_Results_Local(args.resultfile, autoflush=True, verbose=True)

    print("Running %s:%d" % (dataset, args.category))

    if not results.already_calculated(dataset=args.dataset, category=args.category, method=tf_mode+AlphaNet.method_name(), run=args.run):
        data = TextCollectionLoader(dataset=dataset, rep_mode='dense', vectorizer='tf', norm='none', positive_cat=args.category, feat_sel=fs, sublinear_tf=args.sublinear_tf)
        trX, trY = as_variables(data.get_train_set())
        vaX, vaY = as_variables(data.get_validation_set())
        teX, teY = as_variables(data.get_test_set())

        nD, nF = trX.size()
        nC = 1

        model = AlphaNet(input_size=nF, num_classes=nC, hidden=hidden).cuda()

        # Loss and Optimizer (Sigmoid is internally computed.)
        criterion = nn.BCEWithLogitsLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.)

        # Training the Model
        early_stop = EarlyStop(patience=patience)
        for epoch in range(num_epochs):
            trLoss = model.train(trX, trY, optimizer, criterion, batch_size)
            vaAcc, vaLoss = model.test(vaX, vaY, criterion, batch_size)

            print ('Epoch: [%d/%d], Loss: %.8f [valAcc: %.3f %%, valLoss: %.8f ]' % (epoch + 1, num_epochs, trLoss, vaAcc, vaLoss))

            if early_stop.check(vaLoss):
                print("Early stop after %d steps without any improvement in the validation set" % patience)
                break
            else: #shuffle
                perm = torch.randperm(nD).cuda()
                trX = trX[perm]
                trY = trY[perm]

        # Test the Model
        teAcc = model.test(teX,teY,batch_size=batch_size)
        print('Test Accuracy : %.3f %%' % teAcc)

        trX, trY = as_variables(data.get_devel_set())
        trX = F.normalize(trX.data.cpu() * model.linear1.weight.data.cpu(), p=2, dim=1).numpy()
        teX = F.normalize(teX.data.cpu() * model.linear1.weight.data.cpu(), p=2, dim=1).numpy()
        trY = trY.data.cpu().numpy()
        teY = teY.data.cpu().numpy()

        fscore, tp, tn, fp, fn = train_test_svm(trX, trY, teX, teY)
        results.add_row(dataset=args.dataset, category=args.category, method=tf_mode+AlphaNet.method_name(), run=args.run, f1=fscore, tp=tp, tn=tn, fp=fp, fn=fn)

        del trX, teX, trY, teY, model, criterion, optimizer
        gc.collect()

    # ---------------------------------------------------------------------
    for baseline in weight_baselines:
        baseline_name = tf_mode + baseline
        if not results.already_calculated(dataset=args.dataset, category=args.category, method=baseline_name):
            print('\tRunning baseline %s'%baseline)
            data = TextCollectionLoader(dataset=dataset, rep_mode='dense', vectorizer=baseline, norm='l2', positive_cat=args.category, feat_sel=fs, sublinear_tf=args.sublinear_tf)
            trX, trY = data.get_devel_set()
            teX, teY = data.get_test_set()
            fscore, tp, tn, fp, fn = train_test_svm(trX, trY, teX, teY)
            results.add_row(dataset=args.dataset, category=args.category, method=baseline_name, run=0, f1=fscore, tp=tp, tn=tn, fp=fp, fn=fn)

    # collectgarbage();

    # Save the Model
    # torch.save(model.state_dict(), 'model.pkl')
    # print('Checking correlation with information gain...')
    # alpha = model.alpha.data.cpu().numpy()
    # feat_corr_info = get_tsr_statistics(data, information_gain)
    # plt.plot(alpha, feat_corr_info, 'ro')
    # plt.show()
