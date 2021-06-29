#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging

from helper.himputer import NanImputeScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.decomposition import PCA

def import_class(name):
    mod = __import__('.'.join(name.split('.')[:-1]), fromlist=[name.split('.')[-1]])
    return getattr(mod, name.split('.')[-1])

def perform_reduction(X_train, X_test, kind):
    assert len(X_train.shape) >= 2 and len(X_test.shape) >= 2

    # Scale per feature
    new_X_train, new_X_test = np.zeros((X_train.shape[0], X_train.shape[1], 50)), np.zeros((X_test.shape[0], X_test.shape[1], 50))
    for i in range(X_train.shape[1]):
        pca = PCA(n_components=50)
        pca.fit(X_train[:, i, :])
        new_X_train[:, i, :] = pca.transform(X_train[:, i, :])
        new_X_test[:, i, :] = pca.transform(X_test[:, i, :])

    logging.info('applied feature scaling {}'.format(kind))

    return new_X_train, new_X_test

def perform_scaling(X_train, X_test, kind):
    assert len(X_train.shape) >= 2 and len(X_test.shape) >= 2

    # Scale per feature
    for i in range(X_train.shape[2]):
        if kind == 'none':
            return X_train, X_test
        if kind == 'minmax_scaler':
            scaler = MinMaxScaler()
        elif kind == 'standard_scaler':
            scaler = StandardScaler()
        elif kind == 'nan_scaler':
            logging.info('imputed with nan scaler')
            scaler = NanImputeScaler(nan_level=-10)
        else:
            raise NotImplementedError('The scaler {} has not been implemented'.format(kind))
        scaler.fit(X_train[:, :, i])
        X_train[:, :, i] = scaler.transform(X_train[:, :, i])
        X_test[:, :, i] = scaler.transform(X_test[:, :, i])

    logging.info('applied feature scaling {}'.format(kind))

    return X_train, X_test

def perform_oversampling(X, y, kind):
    assert len(X.shape) >= 2

    # Create the oversampler
    oversample = BorderlineSMOTE(kind=kind, random_state=0)
    logging.info('before oversampling {} - {}'.format(kind, [(e, list(y).count(e)) for e in np.unique(y)] if y is not None else []))

    # Flat data for oversampling in 2D
    prev_reshape = X.shape
    X, y = oversample.fit_resample(np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2])), y)
    X = np.reshape(X, (X.shape[0], prev_reshape[1], prev_reshape[2]))
    logging.info('after oversampling {} - {}'.format(kind, [(e, list(y).count(e)) for e in np.unique(y)] if y is not None else []))

    return X, y