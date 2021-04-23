import numpy as np
import pandas as pd

from collections import Counter
from pycm import ConfusionMatrix
from scipy.stats import gmean


def accuracy(tp, fp, fn, tn):
    acc = (tp + tn) / (tp + fp + fn + tn)
    return np.nan_to_num(acc)


def sensitivity(tp, fp, fn, tn):
    tpr = tp / (tp + fn)
    return np.nan_to_num(tpr)


def specificity(tp, fp, fn, tn):
    tnr = tn / (fp + tn)
    return np.nan_to_num(tnr)


def precision(tp, fp, fn, tn):
    tpr = tp / (tp + fp)
    return np.nan_to_num(tpr)


def f1_score(sns, prc):
    f1 = (2 * prc * sns) / (prc + sns)
    return np.nan_to_num(f1)


def balanced_accuracy(sns, spc):
    bac = (sns + spc) / 2
    return bac


def geometric_mean(sns, spc):
    g_mean = np.sqrt(sns * spc)
    return g_mean


def geometric_mean_prc(sns, prc):
    g_mean = np.sqrt(sns * prc)
    return g_mean


def confusion_matrix_scores(cms):
    if len(cms.shape) == 3:
        cms = cms.transpose(1, 2, 0)

    (tn, fp), (fn, tp) = cms

    assert(np.equal.reduce(tn + fp))
    assert(np.equal.reduce(tp + fn))

    if np.any((tn + fp) < (tp + fn)):
        warn("Incorrect negative/positive notation, switching.")
        tn, fp, fn, tp = tp, fn, fp, tn

        if not np.all(tn + fp > tp + fn):
            raise Exception("Could not repair positive/negative notation.")

    acc = accuracy(tp, fp, fn, tn)
    sns = sensitivity(tp, fp, fn, tn)
    spc = specificity(tp, fp, fn, tn)
    prc = precision(tp, fp, fn, tn)
    f1 = f1_score(sns, prc)
    bac = balanced_accuracy(sns, spc)
    gm = geometric_mean(sns, spc)
    gmp = geometric_mean_prc(sns, prc)

    return {
        # 'Accuracy': acc,
        'Sensitivity': sns,
        'Specificity': spc,
        'Precision': prc,
        # 'F1 score': f1,
        "Balanced Accuracy": bac,
        'Gmean_s': gm,
        'Gmean': gmp
    }


def pycm_matrix(confusion_matrix):
    matrix = {
        i: {
            j: int(a)
            for j, a in enumerate(row)
        }
        for i, row in enumerate(confusion_matrix)
    }

    return ConfusionMatrix(matrix=matrix)


def multiclass_matrix_scores(cms):
    dicts = []

    for cm in cms:
        pcm = pycm_matrix(cm)
        results = getattr(pcm, 'overall_stat', {})

        # Add MG-mean
        results['MGM'] = gmean(list(pcm.TPR.values()))
        dicts.append(results)

    result = pd.DataFrame(dicts).to_dict(orient='list')

    return result
