from sklearn.base import ClassifierMixin, BaseEstimator

from .opt import soo


class EUS(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, score_func):
        self.base_estimator = base_estimator
        self.score_func = score_func

        self.res = None
        self.estimator = None

    def fit(self, X, y):
        self.res = soo(X, y, self.base_estimator, self.score_func)
        self.estimator = self.res.opt.get('E')[0, 0]

        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
