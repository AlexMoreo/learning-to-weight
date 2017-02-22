import time
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from utils.metrics import macroF1, microF1
from sklearn.svm import LinearSVC
from data.dataset_loader import TextCollectionLoader
import sys
from data.weighted_vectors import WeightedVectors
from result_table import *


#data = TextCollectionLoader(dataset='reuters21578', vectorizer='tf', rep_mode='sparse')

base = BaselineResultTable('../results/baselines_try_2.csv', classification_mode='binary')

#classifier,weighting,num_features,dataset,category,date,time,elapsedtime,notes,acc,fscore,tp,fp,fn,tn
def check_if_calculated(df, classifier, weighting, num_features, dataset, category):
    query_ = "classifier=='%s' and weighting=='%s' and num_features==%d and dataset=='%s' and category==%d" % (
        classifier, weighting, num_features, dataset, category
    )
    return len(df.query(query_))>0

#q = base.df.query("classifier == 'LinearSVC' and category == 0")

#print len(q)

print check_if_calculated(base.df, "LinearSVC", "tf", 2882, 'reuters21578', 0)
