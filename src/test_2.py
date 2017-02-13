import time

from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from data.dataset_loader import DatasetLoader
from src.utils.metrics import macroF1, microF1
import sys

data = DatasetLoader(dataset='reuters21578', vectorize='tfcw', rep_mode='sparse', feat_sel=0.1)
Xtr, ytr = data.devel_vec, data.devel.target
Xte, yte = data.test_vec, data.test.target

#Xtr=scipy.sparse.csr_matrix(Xtr)
#Xte=scipy.sparse.csr_matrix(Xte)


#Xtr, ytr = data.devel_vec[:300,:500], data.devel.target[:300,:]
#Xte, yte = data.test_vec[:300,:500], data.test.target[:300,:]

tini = time.time()

#rr = RoundRobin(score_func=infogain, k=4000)
#Xtr = rr.fit_transform(Xtr, ytr)
#Xte = rr.transform(Xte)

print time.time()-tini
tini=time.time()

print Xtr.shape
print Xte.shape
print ytr.shape
print yte.shape

# tf ig: 0.567962747193 0.845852017937
# tf gr: 0.600562827435 0.850731142319
# tf conf_weight 0.616643000763 0.858954041204

#for cat in DatasetLoader.valid_catcodes[data.name]:
#    ytr_cat = ytr[:,cat]
bestMacroF1 = 0
bestmicroF1 = 0
param_c = [1e4, 1e3, 1e2, 1e1, 1]
param_loss = ['hinge','squared_hinge']
param_dual = [False, True]
for C in param_c:
    for loss in param_loss:
        for dual in param_dual:
            try:
                svm_ = OneVsRestClassifier(LinearSVC(C=C, loss=loss, dual=dual), n_jobs=-1).fit(Xtr, ytr)
                yte_ = svm_.predict(Xte)

                #svm_ = OneVsRestClassifier(MultinomialNB(), n_jobs=-1).fit(Xtr, ytr)
                #yte_ = svm_.predict(Xte)

                macroF1_ = macroF1(yte, yte_)
                microF1_ = microF1(yte, yte_)

                print C, loss, dual
                print macroF1_
                print microF1_
                print ""
                bestMacroF1 = max(macroF1_, bestMacroF1)
                bestmicroF1 = max(microF1_, bestmicroF1)
            except ValueError:
                pass  # print('Param configuration not supported, skip')

print "Best:", bestMacroF1, bestmicroF1
