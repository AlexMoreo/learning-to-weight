import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import sys
import random
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
    def __init__(self, input_size, num_classes, drop_p=0.25, supervised_statistics=None):
        super(TCClassifierNet, self).__init__()
        #self.linear1 = nn.Linear(input_size, num_classes, bias=True)
        #self.linear2 = nn.Linear(512, num_classes, bias=True)
        self.linear1 = nn.Linear(input_size, input_size/2)
        self.linear2 = nn.Linear(input_size/2, input_size/4)
        self.linear3 = nn.Linear(input_size/4, input_size/8)
        self.linear4 = nn.Linear(input_size/8, num_classes)
        self.training = False
        self.drop_p = drop_p
        self.supervised_statistics = supervised_statistics
        if self.supervised_statistics is not None:
            info_by_feat = self.supervised_statistics.size() [1]
            self.sup_linear1 = nn.Linear(info_by_feat+1, 1)
            # self.sup_linear2 = nn.Linear(100, 1)
            # self.sup_linear3 = nn.Linear(50, 1)

    def supervised_weighting_numpy(self, X, batch_size=50):
        nD,nF = X.size()
        num_batches = nD // batch_size
        if nD % batch_size: num_batches += 1
        XW = np.zeros((nD,nF))
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            XW[start:end] = self.supervised_weighting(X[start:end]).data.cpu().numpy()
        return XW

    def supervised_weighting(self, x):
        if self.supervised_statistics is not None:
            nD,nF = x.size()
            xflat = x.view(nD*nF,1)
            sfeat = self.supervised_statistics.repeat(nD, 1)
            x_feat = torch.cat([xflat,sfeat],1)
            x_h = self.sup_linear1(x_feat)
            # x_h = F.relu(x_h)
            # x_h = F.dropout(x_h, self.drop_p, self.training)
            # x_h = self.sup_linear2(x_h)
            # x_h = F.relu(x_h)
            # x_h = F.dropout(x_h, self.drop_p, self.training)
            # x_h = self.sup_linear3(x_h)
            x_h = x_h.view(nD,nF)
            x_ones = torch.gt(x,0.0).type(torch.FloatTensor).cuda()
            x = x_h * x_ones
        return x

    def forward(self, x):
        self.weighted_documents = self.supervised_weighting(x)
        x = self.linear1(self.weighted_documents)
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, self.training)
        x = self.linear2(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, self.training)
        x = self.linear3(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, self.training)
        x = self.linear4(x)
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
        return 'Torch4'

class EarlyStop:
    def __init__(self, patience=5, low_is_better=True):
        self.best_loss = None
        self.patience = patience
        self.my_patience = patience
        self.has_improved = False
        self.low_is_better = low_is_better

    def check(self, valid_loss):
        if self.best_loss is None:
            self.best_loss = valid_loss
            self.has_improved = True
        elif (self.low_is_better and valid_loss < self.best_loss) or \
                (not self.low_is_better and valid_loss > self.best_loss):
            self.best_loss = valid_loss
            self.my_patience = self.patience
            self.has_improved = True
        else:
            self.my_patience -= 1
            self.has_improved = False
        return self.my_patience == 0

def as_variables(Xy, volatile=True):
    X,y=Xy
    X = Variable(torch.from_numpy(X.astype(float)).float(), requires_grad=False, volatile=volatile).cuda()
    y = Variable(torch.from_numpy(np.squeeze(y).astype(float)).float(), requires_grad=False, volatile=volatile).cuda()
    return X,y


#-----------------------------------------------------------------------------------------------------------------------
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
    parser.add_argument("-m", "--modelpath", help="path to contain the model parameters", type=str, default="../model/TorchClassif.params.dat")
    args = parser.parse_args()


    num_epochs = 100
    learning_rate = 0.01
    batch_size = 128
    patience = 10
    fs = args.fs
    dataset = args.dataset
    weight_baselines = ['tfidf', 'tfchi2', 'tfig', 'tf', 'binary', 'tfrf', 'l1']
    tf_mode = 'Log' if args.sublinear_tf else ''

    torch.backends.cudnn.benchmark = True
    results = Classifier_Results_Local(args.resultfile, autoflush=True, verbose=True)

    print("Running %s:%d" % (dataset, args.category))
    random_seeds=[123456789,234567891,345678912,456789123,567891234,567891234,678912345,789123456,891234567,912345678]
    random_seed = random_seeds[args.run] if args.run != -1 else random.randint(0,100000)

    method_name = TCClassifierNet.method_name() + ('_STW' if args.feat_info else '')
    if args.force or not results.already_calculated(dataset=args.dataset, category=args.category, method=method_name, run=args.run):
        data = TextCollectionLoader(dataset=dataset, rep_mode='dense', vectorizer='tf', norm='none', positive_cat=args.category, feat_sel=fs, sublinear_tf=False, random_state=random_seed)
        nD = data.num_devel_documents()
        m = None
        if args.feat_info:
            m = Variable(torch.from_numpy(np.array(
                [[x.tpr(), x.fpr(),
                  x.p_f(), x.p_tp(), x.p_fp(), x.p_tn(), x.p_fn(),
                  information_gain(x), idf(x), relevance_frequency(x), gss(x), conf_weight(x), pointwise_mutual_information(x), chi_square(x)
                  ] for x in np.squeeze(data.get_4cell_matrix())],
                dtype=np.float32)), requires_grad=False, volatile=False).cuda()
        trX, trY = as_variables(data.get_train_set(), volatile=False)
        vaX, vaY = as_variables(data.get_validation_set())
        teX, teY = as_variables(data.get_test_set())

        nD, nF = trX.size()
        nC = 1
        print("nD={}, nF={}, pC={}".format(nD,nF,data.train_class_prevalence(0)))

        model = TCClassifierNet(input_size=nF, num_classes=nC, supervised_statistics=m).cuda()

        # Loss and Optimizer (Sigmoid is internally computed.)
        criterion = nn.BCEWithLogitsLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Training the Model
        early_stop = EarlyStop(patience=patience, low_is_better=False)
        for epoch in range(num_epochs):
            trLoss = model.train(trX, trY, optimizer, criterion, batch_size)
            vaY_, vaLoss = model.test(vaX, vaY, criterion, batch_size)
            fscore_val = f1(single_metric_statistics(vaY.data.cpu().numpy(), vaY_))

            teY_, teLoss = model.test(teX, teY, criterion, batch_size)
            fscore_test = f1(single_metric_statistics(teY.data.cpu().numpy(), teY_))
            print ('Epoch: [%d/%d], Loss: %.8f [vaLoss=%.8f vaF1=%.4f teF1=%.4f]' % (epoch + 1, num_epochs, trLoss, vaLoss, fscore_val, fscore_test))

            if early_stop.check(fscore_val):
                print("Early stop after %d steps without any improvement in the validation set" % patience)
                break
            else: #shuffle
                perm = torch.randperm(nD).cuda()
                trX = trX[perm]
                trY = trY[perm]

            if early_stop.has_improved:
                torch.save(model.state_dict(), args.modelpath)

        # Test the Model
        model.load_state_dict(torch.load(args.modelpath))
        teY_ = model.test(teX,teY,batch_size=batch_size)
        cell = single_metric_statistics(teY.data.cpu().numpy(), teY_)
        fscore = f1(cell)
        print('Test F1: %.3f %%' % fscore)
        results.add_row(dataset=args.dataset, category=args.category, method=method_name, run=args.run, f1=fscore, tp=cell.tp, tn=cell.tn, fp=cell.fp, fn=cell.fn)

        trX, trY = as_variables(data.get_devel_set())
        trX = model.supervised_weighting_numpy(trX)
        teX = model.supervised_weighting_numpy(teX)
        trY = trY.data.cpu().numpy()
        teY = teY.data.cpu().numpy()
        fscore, tp, tn, fp, fn = train_test_svm(trX, trY, teX, teY)
        results.add_row(dataset=args.dataset, category=args.category, method=method_name+"_SVM",
                        run=args.run, f1=fscore, tp=tp, tn=tn, fp=fp, fn=fn)

        del trX, teX, trY, teY, model, criterion, optimizer
        gc.collect()

    # ------------------------------------------------------------------------------------------------------------------
    for baseline in weight_baselines:
        baseline_name = tf_mode + baseline if baseline!='l1' else baseline
        if not results.already_calculated(dataset=args.dataset, category=args.category, method=baseline_name):
            print('\tRunning baseline %s'%baseline)
            data = TextCollectionLoader(dataset=dataset, rep_mode='dense', vectorizer=baseline, norm='l2' if baseline!='l1' else 'none', positive_cat=args.category, feat_sel=fs,
                                        sublinear_tf=args.sublinear_tf if baseline!='l1' else False)
            trX, trY = data.get_devel_set()
            teX, teY = data.get_test_set()
            fscore, tp, tn, fp, fn = train_test_svm(trX, trY, teX, teY)
            results.add_row(dataset=args.dataset, category=args.category, method=baseline_name, run=0, f1=fscore, tp=tp, tn=tn, fp=fp, fn=fn)
