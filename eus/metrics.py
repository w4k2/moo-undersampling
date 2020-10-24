from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy.stats import gmean
import numpy as np


def sensitivity(tp, fp, fn, tn):
    tpr = tp / (tp + fn)
    return np.nan_to_num(tpr)


def specificity(tp, fp, fn, tn):
    tnr = tn / (fp + tn)
    return np.nan_to_num(tnr)


def precision(tp, fp, fn, tn):
    tpr = tp / (tp + fp)
    return np.nan_to_num(tpr)


def soo_score_bac(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    (tn, fp), (fn, tp) = cm
    sns = sensitivity(tp, fp, fn, tn)
    spc = specificity(tp, fp, fn, tn)
    return np.mean(sns, spc)


def soo_score_gmean1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    (tn, fp), (fn, tp) = cm
    sns = sensitivity(tp, fp, fn, tn)
    spc = specificity(tp, fp, fn, tn)
    return gmean((sns, spc))


def soo_score_gmean2(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    (tn, fp), (fn, tp) = cm
    sns = sensitivity(tp, fp, fn, tn)
    ppv = precision(tp, fp, fn, tn)
    return gmean((sns, ppv))


def soo_score_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def moo_score_sns_spc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    (tn, fp), (fn, tp) = cm
    return [
        sensitivity(tp, fp, fn, tn),
        specificity(tp, fp, fn, tn)
    ]

def moo_score_sns_ppv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    (tn, fp), (fn, tp) = cm
    return [
        sensitivity(tp, fp, fn, tn),
        precision(tp, fp, fn, tn)
    ]

def moo_score_sns_spc_ppv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    (tn, fp), (fn, tp) = cm
    return [
        sensitivity(tp, fp, fn, tn),
        specificity(tp, fp, fn, tn),
        precision(tp, fp, fn, tn),
    ]
