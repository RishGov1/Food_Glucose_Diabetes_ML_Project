import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class Persistence(RegressorMixin, BaseEstimator):
    def __init__(self, column=0):
        self.column = column

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.repeat(np.asarray(X)[:, self.column:self.column+1], 2, axis=1)


class ResidualModel(RegressorMixin, BaseEstimator):
    def __init__(self, estimator, column=0):
        self.estimator = estimator
        self.column = column

    def fit(self, X, y):
        X = np.asarray(X)
        self.model_ = clone(self.estimator).fit(X, np.asarray(y) - X[:, self.column:self.column+1])
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.model_.predict(X) + X[:, self.column:self.column+1]
