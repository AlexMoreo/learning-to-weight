import time
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from utils.metrics import macroF1, microF1
from sklearn.svm import LinearSVC
from data.dataset_loader import DatasetLoader
import sys
from data.weighted_vectors import WeightedVectors
from result_table import *


data = DatasetLoader(dataset='reuters21578', vectorizer='tf', rep_mode='sparse')

base = BaselineResultTable('../results/baselines.csv')
base.init_row_result(classifier_name='clasificador base', data=data)
base.commit()
base = Learning2Weight_ResultTable('../results/learnt.csv')
base.init_row_result(classifier_name='learned 1A', data=data)
base.set_learn_params(1000, 12345, True,True,False)
base.commit()
sys.exit()



#data = WeightedVectors.unpickle('../vectors/', 'reu_Cmulticlass_FS0.10_H100_lr0.00100_Oadam_NTrue_nTrue_Poff_R0.pickle')

Xtr, ytr = data.get_devel_set()
tini = time.time()
print Xtr.shape
print ytr.shape

# tf ig: 0.567962747193 0.845852017937
# tf gr: 0.600562827435 0.850731142319 - .60909768332 .849739473314 (with GridSearchCV)
# tf conf_weight 0.616643000763 0.858954041204

#0.617121197114
#{'estimator__dual': True, 'estimator__C': 1, 'estimator__loss': 'hinge'}
#MacroF1 test:  0.609662203524
#MicroF1 test:  0.859154929577
parameters = {'estimator__C':[1e4, 1e3, 1e2, 1e1, 1], 'estimator__loss':['hinge','squared_hinge'], 'estimator__dual':[False, True]}
svm_ = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
model_tunning = GridSearchCV(svm_, param_grid=parameters,
                             scoring=make_scorer(macroF1),
                             error_score=0, refit=True, n_jobs=-1)
tunned = model_tunning.fit(Xtr, ytr)

print model_tunning.best_score_
print model_tunning.best_params_

Xte, yte = data.get_test_set()
yte_ = tunned.predict(Xte)

print "MacroF1 test: ", macroF1(yte, yte_)
print "MicroF1 test: ", microF1(yte, yte_)

sys.exit()

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
