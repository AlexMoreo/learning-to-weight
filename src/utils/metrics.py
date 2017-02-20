from feature_selection.tsr_function import ContTable
import numpy as np

"""
Scikit learn provides a full set of evaluation metrics, but they treat special cases differently.
I.e., when the number of true positives, false positives, and false negatives ammount to 0, all
affected metrices (precision, recall, and thus f1) output 0 in Scikit learn.
We adhere to the common practice of outputting 1 in this case since the classifier has correctly
classified all examples as negatives.
"""

def accuracy(cell):
    return cell.tp*1.0 / (cell.tp + cell.fp + cell.fn + cell.tn)

def f1(cell):
    num = 2.0 * cell.tp
    den = 2.0 * cell.tp + cell.fp + cell.fn
    if den>0: return num / den
    #we define f1 to be 1 if den==0 since the classifier has correctly classified all instances as negative
    return 1.0

#true_labels and predicted_labels are two vectors of shape (number_documents,)
def single_metric_statistics(true_labels, predicted_labels):
    tp = sum([1 for x,y in zip(true_labels,predicted_labels) if x==y==1])
    fp = sum([1 for x, y in zip(true_labels, predicted_labels) if x == 0 and y == 1])
    fn = sum([1 for x, y in zip(true_labels, predicted_labels) if x == 1 and y == 0])
    tn = sum([1 for x, y in zip(true_labels, predicted_labels) if x == 0 and y == 0])
    if tp+fp+fn+tn != len(true_labels): raise ValueError("Format not consistent between true and predicted labels.")
    return ContTable(tp=tp, tn=tn, fp=fp, fn=fn)

#if the classifier is single class, then the prediction is a vector of shape=(nD,) which causes issues when compared
#to the true labels (of shape=(nD,1)). This method increases the dimensions of the predictions.
def __check_consistency_and_adapt(true_labels, predictions):
    if predictions.ndim == 1:
        return __check_consistency_and_adapt(true_labels, np.expand_dims(predictions, axis=1))
    if true_labels.ndim == 1:
        return __check_consistency_and_adapt(np.expand_dims(true_labels, axis=1),predictions)
    if true_labels.shape != predictions.shape:
        raise ValueError("True and predicted label matrices shapes are inconsistent %s %s."
                         % (true_labels.shape, predictions.shape))
    _,nC = true_labels.shape
    return true_labels, predictions, nC

#true_labels and predicted_labels are two matrices in sklearn.preprocessing.MultiLabelBinarizer format
def macroF1(true_labels, predicted_labels):
    true_labels, predicted_labels, nC = __check_consistency_and_adapt(true_labels, predicted_labels)

    macrof1 = 0.0
    for c in range(nC):
        macrof1 += f1(single_metric_statistics(true_labels[:,c], predicted_labels[:,c]))

    return macrof1/nC

#true_labels and predicted_labels are two matrices in sklearn.preprocessing.MultiLabelBinarizer format
def microF1(true_labels, predicted_labels):
    true_labels, predicted_labels, nC = __check_consistency_and_adapt(true_labels, predicted_labels)

    def aggregate_cell(accum, other):
        accum.tp+=other.tp
        accum.fp+=other.fp
        accum.fn+=other.fn
        accum.tn+=other.tn

    accum = ContTable()
    for c in range(nC):
        aggregate_cell(accum, single_metric_statistics(true_labels[:, c], predicted_labels[:, c]))

    return f1(accum)

