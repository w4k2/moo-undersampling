from sklearn.base import ClassifierMixin, BaseEstimator
import mcdm
import numpy as np

from .opt import moo


class MEUS(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, score_func, n_metrics):
        self.base_estimator = base_estimator
        self.score_func = score_func
        self.n_metrics = n_metrics

        self.res = None
        self.estimator = None

    def fit(self, X, y):
        self.res = moo(X, y, self.base_estimator, self.score_func, self.n_metrics)

        # Select MCDM solution
        scores = -1 * self.res.opt.get('F')
        u_scores = np.unique(scores, axis=0)
        best = int(mcdm.rank(u_scores, s_method='SAW')[0][0][1:]) - 1
        e_idx = np.where(scores == u_scores[best])[0][0]

        self.estimator = self.res.opt.get('E')[e_idx, 0]

        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
