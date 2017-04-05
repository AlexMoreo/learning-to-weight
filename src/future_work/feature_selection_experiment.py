def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os, sys
from data.dataset_loader import TextCollectionLoader
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from utils.metrics import *
import matplotlib.pyplot as plt
from feature_selection.round_robin import RoundRobin
import time
import operator
from feature_selection.tsr_function import *

def fit_model_hyperparameters(data, parameters, model):
    parameters = {'estimator__' + key: parameters[key] for key in parameters.keys()}
    model = OneVsRestClassifier(model, n_jobs=-1)
    model_tunning = GridSearchCV(model, param_grid=parameters,
                                 scoring=make_scorer(macroF1), error_score=0, refit=True, cv=5, n_jobs=-1)

    Xtr, ytr = data.get_devel_set()
    Xtr.sort_indices()

    tunned = model_tunning.fit(Xtr, ytr)

    return tunned


def fit_and_test_model(data, parameters, model):
    init_time = time.time()
    tunned_model = fit_model_hyperparameters(data, parameters, model)
    tunning_time = time.time() - init_time
    print("%s: best parameters %s, best score %.3f, took %.3f seconds" %
          (type(model).__name__, tunned_model.best_params_, tunned_model.best_score_, tunning_time))

    Xte, yte = data.get_test_set()
    Xte.sort_indices()
    yte_ = tunned_model.predict(Xte)

    macro_f1 = macroF1(yte, yte_)
    micro_f1 = microF1(yte, yte_)
    print("Test scores: %.3f macro-f1, %.3f micro-f1" % (macro_f1, micro_f1))

    return macro_f1, micro_f1


def linear_svm(data):
    parameters = {'C': [1e2, 1e1, 1, 1e-1],
                  'loss': ['hinge', 'squared_hinge'],
                  'dual': [True, False]}
    model = LinearSVC()
    return fit_and_test_model(data, parameters, model)

def get_onehot(dimensions, one_pos):
    one_hot = np.zeros(dimensions, dtype=np.int64)
    if one_pos:
        one_hot[one_pos]=1
    return one_hot

def naive_equivalent_class_encoding(true_labels):
    eqclass_encodding = {}
    for document_labels in true_labels:
        eqc_hash = np.array_str(document_labels)
        if eqc_hash not in eqclass_encodding:
            eqclass_encodding[eqc_hash] = len(eqclass_encodding)

    n_eqclasses = len(eqclass_encodding)
    print n_eqclasses,"different equivalent classes from",len(true_labels[0]),"original classes"

    mod_labels = []
    for document_labels in true_labels:
        eqc_hash = np.array_str(document_labels)
        mod_labels.append(get_onehot(n_eqclasses, eqclass_encodding[eqc_hash]))
    return np.array(mod_labels)

def frequency_equivalent_class_encoding(true_labels):
    eqclass_encodding_freq = {}
    for document_labels in true_labels:
        eqc_hash = np.array_str(document_labels)
        if eqc_hash not in eqclass_encodding_freq:
            eqclass_encodding_freq[eqc_hash] = 0
        eqclass_encodding_freq[eqc_hash] += 1

    sorted_by_freq = sorted(eqclass_encodding_freq.items(), key=operator.itemgetter(1), reverse=True)
    frequent = [(c,freq) for c,freq in sorted_by_freq if freq > 1]
    infrequent = [(c,freq) for c,freq in sorted_by_freq if freq == 1]

    eqclass_encodding_freq = {}
    # indexes all classes of equivalence with a distinct index
    for c,_ in frequent:
        eqclass_encodding_freq[c] = len(eqclass_encodding_freq)

    # groups the infrequent classes into a single id
    n_eqclasses = len(eqclass_encodding_freq) + 1
    infrequent_dimnesion = len(eqclass_encodding_freq)
    for c,_ in infrequent:
        eqclass_encodding_freq[c] = infrequent_dimnesion

    print n_eqclasses,"different equivalent classes from",len(true_labels[0]),"original classes"

    mod_labels = []
    for document_labels in true_labels:
        eqc_hash = np.array_str(document_labels)
        mod_labels.append(get_onehot(n_eqclasses, eqclass_encodding_freq[eqc_hash]))
    return np.array(mod_labels)


def frequency_classpreservation_equivalent_class_encoding(true_labels):
    eqclass_encodding_freq = {}
    for document_labels in true_labels:
        eqc_hash = np.array_str(document_labels)
        if eqc_hash not in eqclass_encodding_freq:
            eqclass_encodding_freq[eqc_hash] = 0
        eqclass_encodding_freq[eqc_hash] += 1

    sorted_by_freq = sorted(eqclass_encodding_freq.items(), key=operator.itemgetter(1), reverse=True)
    frequent = [(c,freq) for c,freq in sorted_by_freq if freq > 1]
    infrequent = [(c,freq) for c,freq in sorted_by_freq if freq == 1]

    eqclass_encodding_freq = {}
    # indexes all classes of equivalence with a distinct index
    for c,_ in frequent:
        eqclass_encodding_freq[c] = len(eqclass_encodding_freq)

    # groups the infrequent classes into a single id
    n_eqclasses = len(eqclass_encodding_freq) + 1
    infrequent_dimnesion = len(eqclass_encodding_freq)
    for c,_ in infrequent:
        eqclass_encodding_freq[c] = infrequent_dimnesion

    print n_eqclasses,"different equivalent classes from",len(true_labels[0]),"original classes"

    mod_labels = []
    for document_labels in true_labels:
        eqc_hash = np.array_str(document_labels)
        ec_vector = get_onehot(n_eqclasses, eqclass_encodding_freq[eqc_hash])
        mod_labels.append(np.concatenate((document_labels, ec_vector)))
    return np.array(mod_labels)

def frequency_classpreservation_clean_equivalent_class_encoding(true_labels):
    eqclass_encodding_freq = {}
    for document_labels in true_labels:
        eqc_hash = np.array_str(document_labels)
        if eqc_hash not in eqclass_encodding_freq:
            eqclass_encodding_freq[eqc_hash] = 0
        #only count frequency for those equivalent classes which involve more than one category
        n_related_categories = np.sum(document_labels)
        if n_related_categories > 1:
            eqclass_encodding_freq[eqc_hash] += 1

    sorted_by_freq = sorted(eqclass_encodding_freq.items(), key=operator.itemgetter(1), reverse=True)

    # introducing a frequency threshold
    freq_threshold = 1
    frequent = [(c,freq) for c,freq in sorted_by_freq if freq > freq_threshold]
    infrequent = [(c,freq) for c,freq in sorted_by_freq if freq <= freq_threshold]

    eqclass_encodding_freq = {}
    # indexes all classes of equivalence with a distinct index
    for c,_ in frequent:
        eqclass_encodding_freq[c] = len(eqclass_encodding_freq)

    # groups the infrequent classes into a single id, infrequent eq-classes are dropped out
    n_eqclasses = len(eqclass_encodding_freq)
    for c,_ in infrequent:
        eqclass_encodding_freq[c] = None # drop

    print n_eqclasses,"different equivalent classes from",len(true_labels[0]),"original classes"

    mod_labels = []
    for document_labels in true_labels:
        eqc_hash = np.array_str(document_labels)
        ec_vector = get_onehot(n_eqclasses, eqclass_encodding_freq[eqc_hash])
        mod_labels.append(np.concatenate((document_labels, ec_vector)))
    return np.array(mod_labels)


#------------------------------------------------------------------------------------------------------------------------

def read_result_file(path):
    ratio_macro_micro = []
    for line in open(path, 'r').readlines():
        ratio_macro_micro.append([float(x) for x in line.split()])
    return zip(*ratio_macro_micro)

def check_ratios_agreement(ratios, fs_ratios):
    for i in range(len(fs_ratios)):
        if fs_ratios[i] != ratios[i]:
            print "Ratios", ratios, "do not coincide with expected", fs_ratios
            sys.exit()
    return True


def plot_fs_roundrobin(dataset, fs_ratios, resultfile, tsr_function):
    if os.path.exists(resultfile):
        print "Reading pre-calculated results from",resultfile
        ratios, macro, micro = read_result_file(resultfile)
        if check_ratios_agreement(ratios, fs_ratios):
            return macro, micro

    with open(resultfile, 'w') as results:
        fs_ratios.sort()
        result_series = []
        for i,ratio in enumerate(fs_ratios):
            data = TextCollectionLoader(dataset=dataset, feat_sel=ratio, tsr_function=tsr_function)
            macro_f1, micro_f1 = linear_svm(data)
            results.write(resultpath + + str(ratio) + "\t" + str(macro_f1) + "\t" + str(micro_f1) + "\n")
            result_series.append((macro_f1, micro_f1))

    return zip(*result_series)

def plot_fs_roundrobin_eqclasses(dataset, fs_ratios, resultfile, tsr_function, eq_class_method=naive_equivalent_class_encoding):
    if os.path.exists(resultfile):
        print "Reading pre-calculated results from",resultfile
        ratios, macro, micro = read_result_file(resultfile)
        if check_ratios_agreement(ratios, fs_ratios):
            return macro, micro

    with open(resultfile, 'w') as results:
        data = TextCollectionLoader(dataset=dataset)
        original_classification = data.devel.target
        ranked_features_pickle = resultpath + dataset+'_'+tsr_function.__name__+'_'+eq_class_method.__name__+".pickle"
        eqclass_classification = eq_class_method(original_classification)

        fs_ratios.sort()
        result_series = []
        for i,ratio in enumerate(fs_ratios):
            print "Ratio",ratio,"completed",(i+1),'/',len(fs_ratios)
            data.devel.target = eqclass_classification
            data.feature_selection(feat_sel=ratio, score_func=tsr_function, features_rank_pickle_path=ranked_features_pickle)
            data.devel.target = original_classification

            macro_f1, micro_f1 = linear_svm(data)
            results.write(resultpath +str(ratio)+"\t"+str(macro_f1)+"\t"+str(micro_f1)+"\n")
            result_series.append((macro_f1, micro_f1))

            #reload data
            if i < len(fs_ratios)-1:
                data = TextCollectionLoader(dataset=dataset)

    return zip(*result_series)


resultpath = "../results/futurework/feature_selection/"
dataset = "reuters21578"
feat_sel = [0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

tsr_function = posneg_information_gain
#tsr_function = gss
result_prefix = dataset+"_"+tsr_function.__name__
orig_macro_f1, orig_micro_f1 = plot_fs_roundrobin(dataset=dataset,fs_ratios=feat_sel, tsr_function=tsr_function,
                                                  resultfile=result_prefix+"_RR.results")
eq_macro_f1, eq_micro_f1 = plot_fs_roundrobin_eqclasses(dataset=dataset, fs_ratios=feat_sel,tsr_function=tsr_function,
                                                        resultfile=result_prefix+"_naiveEC.results", eq_class_method=naive_equivalent_class_encoding)
fr_eq_macro_f1, fr_eq_micro_f1 = plot_fs_roundrobin_eqclasses(dataset=dataset, fs_ratios=feat_sel,tsr_function=tsr_function,
                                                        resultfile=result_prefix+"_freq_EC.results", eq_class_method=frequency_equivalent_class_encoding)
frp_eq_macro_f1, frp_eq_micro_f1 = plot_fs_roundrobin_eqclasses(dataset=dataset, fs_ratios=feat_sel,tsr_function=tsr_function,
                                                        resultfile=result_prefix+"_freqP_EC.results", eq_class_method=frequency_classpreservation_equivalent_class_encoding)
cleanfrp_eq_macro_f1, cleanfrp_eq_micro_f1 = plot_fs_roundrobin_eqclasses(dataset=dataset, fs_ratios=feat_sel,tsr_function=tsr_function,
                                                        resultfile=result_prefix+"_freqPClean_EC.results", eq_class_method=frequency_classpreservation_clean_equivalent_class_encoding)


x_axis = feat_sel

plot_orig_MF1,_ = plt.plot(x_axis, orig_macro_f1, 'ro', x_axis, orig_macro_f1, 'r-', label='RR Macro-F1')
plot_orig_mF1,_ = plt.plot(x_axis, orig_micro_f1, 'r^', x_axis, orig_micro_f1, 'r-', label='RR micro-F1')
plot_eqc_MF1,_ = plt.plot(x_axis, eq_macro_f1, 'g--', x_axis, eq_macro_f1, 'g-', label='ec(RR) Macro-F1')
plot_eqc_mF1,_ = plt.plot(x_axis, eq_micro_f1, 'g^', x_axis, eq_micro_f1, 'g-', label='ec(RR) micro-F1')
plot_feqc_MF1,_ = plt.plot(x_axis, fr_eq_macro_f1, 'b--', x_axis, fr_eq_macro_f1, 'b-', label='fec(RR) Macro-F1')
plot_feqc_mF1,_ = plt.plot(x_axis, fr_eq_micro_f1, 'b^', x_axis, fr_eq_micro_f1, 'b-', label='fec(RR) micro-F1')
plot_feqp_MF1,_ = plt.plot(x_axis, frp_eq_macro_f1, 'ks', x_axis, frp_eq_macro_f1, 'k-', label='fep(RR) Macro-F1')
plot_feqp_mF1,_ = plt.plot(x_axis, frp_eq_micro_f1, 'ko', x_axis, frp_eq_micro_f1, 'k-', label='fep(RR) micro-F1')
plot_feqpc_MF1,_ = plt.plot(x_axis, cleanfrp_eq_macro_f1, 'ms', x_axis, cleanfrp_eq_macro_f1, 'm-', label='fepc(RR) Macro-F1')
plot_feqpc_mF1,_ = plt.plot(x_axis, cleanfrp_eq_micro_f1, 'mo', x_axis, cleanfrp_eq_micro_f1, 'm-', label='fepc(RR) micro-F1')

plt.legend([plot_orig_MF1, plot_orig_mF1, plot_eqc_MF1, plot_eqc_mF1, plot_feqc_MF1, plot_feqc_mF1, plot_feqp_MF1, plot_feqp_mF1, plot_feqpc_MF1, plot_feqpc_mF1]),
           #['RR Macro-F1', 'RR micro-F1', 'ec-RR Macro-F1', 'ec-RR micro-F1', 'fec-RR Macro-F1', 'fec-RR micro-F1', 'fep-RR Macro-F1', 'fep-RR micro-F1', 'fepc-RR Macro-F1', 'fepc-RR micro-F1'])
box = plt.subplot(111).get_position()
plt.subplot(111).set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
#plt.legend(loc=0, borderaxespad=0.)
plt.title(dataset.title()+' '+tsr_function.__name__.title())
plt.ylabel('F1')
plt.xlabel('selection ratio')
plt.grid(True)
plt.savefig(resultpath +'plot'+dataset+tsr_function.__name__+'.pdf', format='PDF')
plt.show()


