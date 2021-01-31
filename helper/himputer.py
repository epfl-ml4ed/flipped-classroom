from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np

class NanImputeScaler(BaseEstimator, TransformerMixin):
    """Scale an array with missing values, then impute them
    with a dummy value. This prevents the imputed value from impacting
    the mean/standard deviation computation during scaling.

    Parameters
    ----------
    with_mean : bool, optional (default=True)
        Whether to center the variables.

    with_std : bool, optional (default=True)
        Whether to divide by the standard deviation.

    nan_level : int or float, optional (default=-99.)
        The value to impute over NaN values after scaling the other features.
    """
    def __init__(self, with_mean=True, with_std=True, nan_level=-1.):
        self.with_mean = with_mean
        self.with_std = with_std
        self.nan_level = nan_level

    def fit(self, X, y=None):
        # Check the input array, but don't force everything to be finite.
        # This also ensures the array is 2D
        X = check_array(X, force_all_finite=False, ensure_2d=True)

        # compute the statistics on the data irrespective of NaN values
        self.means_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        return self

    def transform(self, X):
        # Check that we have already fit this transformer
        check_is_fitted(self, "means_")

        # get a copy of X so we can change it in place
        X = check_array(X, force_all_finite=False, ensure_2d=True)

        # center if needed
        if self.with_mean:
            X -= self.means_
        # scale if needed
        if self.with_std:
            X /= self.std_

        # now fill in the missing values
        X[np.isnan(X)] = self.nan_level
        return X