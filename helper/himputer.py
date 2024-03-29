from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np

class NanImputeScaler(BaseEstimator, TransformerMixin):

    def __init__(self, with_mean=True, with_std=True, nan_level=-1.):
        self.with_mean = with_mean
        self.with_std = with_std
        self.nan_level = nan_level

    def fit(self, X, y=None):

        X = check_array(X, force_all_finite=False, ensure_2d=True)

        # compute the statistics on the data irrespective of NaN values
        self.means_ = np.nanmean(X)
        self.std_ = np.nanstd(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "means_")

        X = check_array(X, force_all_finite=False, ensure_2d=True)

        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])

        if self.with_mean and self.means_ != 0:
            X -= self.means_

        if self.with_std and self.std_ != 0:
            X /= self.std_

        return X