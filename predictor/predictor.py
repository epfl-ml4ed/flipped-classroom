#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import pickle
import time
import json
import os

class Predictor():

    def __init__(self, name=None):
        self.name = name
        self.time = time.strftime('%Y%m%d_%H%M%S')

    def save(self, settings):
        assert self.predictor is not None and self.predictor is not None and settings['workdir'].endswith('/')
        str_f1 = str(np.round(np.mean([v[settings['scoring'] if 'scoring' in settings else 'f1'] for v in self.stats]), 2)).replace('.', '')
        if not os.path.exists(os.path.join(settings['workdir'], 'predictor', self.time + '-' + self.name + '-' + str_f1)):
            os.makedirs(os.path.join(settings['workdir'], 'predictor', self.time + '-' + self.name + '-' + str_f1))
        with open(os.path.join(settings['workdir'], 'predictor', self.time + '-' + self.name + '-' + str_f1, 'model.h5'), 'wb') as file:
            pickle.dump(self.predictor, file)
        with open(os.path.join(settings['workdir'], 'predictor', self.time + '-' + self.name + '-' + str_f1, 'params.txt'), 'w') as file:
            file.write(json.dumps(settings))
        with open(os.path.join(settings['workdir'], 'predictor', self.time + '-' + self.name + '-' + str_f1, 'stats.txt'), 'w') as file:
            file.write(json.dumps(self.stats))

    def load(self, settings):
        assert os.path.join(settings['workdir']) is not None
        self.predictor = pickle.load(open(os.path.join(settings['workdir'], 'model.h5'), 'rb'))

    def build(self, settings):
        self.predictor = None

    def compile(self, settings):
        pass

    def fit(self, X, y, settings):
        self.predictor.fit(X if len(X.shape) <=2 else np.average(X, axis=1), y)

    def __str__(self):
        return 'Name: {}'.format(self.name)

    def train(self, X, y, settings):
        assert self.predictor is not None
        kf = StratifiedKFold(n_splits=settings['folds'])
        self.stats = [None for _ in np.arange(settings['folds'])]
        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pass_ix = np.where(y_test==0)[0]
            fail_ix = np.where(y_test==1)[0]
            self.fit(X_train, y_train, settings)
            self.stats[fold] = self.evaluate(X_test, y_test, {**settings, **{'pass_ix': pass_ix, 'fail_ix': fail_ix}})
        self.save(settings)
        return self.stats

    def predict(self, X, settings, proba=False):
        assert self.predictor is not None
        if proba:
            return self.predictor.predict_proba(X if len(X.shape) <= 2 else np.average(X, axis=1))
        return np.round(self.predictor.predict(X if len(X.shape) <= 2 else np.average(X, axis=1)))

    def build_with_grid(self, settings):
        assert self.predictor is not None
        self.predictor = GridSearchCV(self.predictor, settings['grid'], cv=settings['cv'], scoring=settings['scoring'])

    def evaluate(self, X, y, settings):
        y_pred = self.predict(X, settings)
        stats = {}
        stats['bacc'] = metrics.balanced_accuracy_score(y, y_pred)
        stats['f1'] = metrics.f1_score(y, y_pred)
        if 'fail_ix' in settings and 'pass_ix' in settings:
            stats['rfail'] = np.sum(y_pred[settings['fail_ix']]) / len(y[settings['fail_ix']])
            stats['rpass'] = 1 - np.sum(y_pred[settings['pass_ix']]) / len(y[settings['pass_ix']])
        fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=1)
        stats['auc'] = metrics.auc(fpr, tpr)
        return stats
