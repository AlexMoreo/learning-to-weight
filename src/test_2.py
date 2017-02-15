import time

from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from data.dataset_loader import DatasetLoader
from utils.metrics import macroF1, microF1
import sys
from sklearn.model_selection import GridSearchCV


data = DatasetLoader(dataset='reuters21578', vectorize='tfgr', rep_mode='sparse', feat_sel=0.1)
Xtr, ytr = data.get_devel_set()
tini = time.time()
print Xtr.shape
print ytr.shape

# tf ig: 0.567962747193 0.845852017937
# tf gr: 0.600562827435 0.850731142319
# tf conf_weight 0.616643000763 0.858954041204

#parameters = {'estimator__C':[1e4, 1e3, 1e2, 1e1, 1], 'estimator__loss':['hinge','squared_hinge'], 'estimator__dual':[False, True]}
#svm_ = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
#model_tunning = GridSearchCV(svm_, param_grid=parameters,
                             #scoring=metrics.f1_score,
#                             error_score=0, refit=True, n_jobs=-1)
#model_tunning.fit(Xtr, ytr)

#print model_tunning.best_score_
#print model_tunning.best_params_

#sys.exit()

nC = ytr.shape[1]
bestMacroF1 = 0
bestmicroF1 = 0
#param_c = [1e4, 1e3, 1e2, 1e1, 1]
param_c = [1]
#param_loss = ['hinge','squared_hinge']
param_loss = ['squared_hinge']
param_dual = [True]
#param_dual = [False, True]

Xtr, ytr = data.get_devel_set()
Xte, yte = data.get_test_set()
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
                if macroF1_ > bestMacroF1:
                    bestMacroF1 = max(macroF1_, bestMacroF1)
                    bestmicroF1 = max(microF1_, bestmicroF1)


            except ValueError as e:
                pass #print "ValueError: %s" % (e.message)

print "Best:", bestMacroF1, bestmicroF1
