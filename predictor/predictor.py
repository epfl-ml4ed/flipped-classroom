#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pandas as pd
import numpy as np
import logging
import pickle
import time
import json
import os

class Predictor():

    def __init__(self, name=None):
        self.name = name
        self.time = time.strftime('%Y%m%d_%H%M%S')
        self.type = 'sklearn'
        self.depth = 'shallow'

    def isdepth(self, depth):
        return depth == 'all' or self.depth == depth

    def save(self, settings):
        assert self.predictor is not None and self.predictor is not None and 'target_type' in settings

        # Set metric to show in the folder name
        if settings['target_type'] == 'classification':
            str_metric = str(np.round(np.mean([v[settings['scoring'] if 'scoring' in settings else 'f1'] for v in self.stats]), 2)).replace('.', '')
        else:
            str_metric = str(np.round(np.mean([v[settings['scoring'] if 'scoring' in settings else 'mse'] for v in self.stats]), 2)).replace('.', '')

        filename = settings['timeframe'] + '-' + settings['course_id'].replace('-', '_') + '-' + settings['target'].replace('-', '_') + '-' + settings['model'].split('.')[-2] + '-' + settings['feature_set'].split('-')[1] + '-' + self.time + '-' + str_metric

        # Create the predictor main directory
        if not os.path.exists(os.path.join(settings['workdir'], filename)):
            os.makedirs(os.path.join(settings['workdir'], filename))
        # Save tehe trained model
        if self.type == 'sklearn':
            with open(os.path.join(settings['workdir'], filename, 'model.h5'), 'wb') as file:
                pickle.dump(self.predictor, file)
        else:
            self.predictor.save(os.path.join(settings['workdir'], filename, 'model.h5'))
        # Save train and test parameters
        with open(os.path.join(settings['workdir'], filename, 'params.txt'), 'w') as file:
            file.write(json.dumps(settings))
        # Save evaluation metrics
        stats = pd.DataFrame(self.stats)
        stats.index.name = 'fold'
        stats.to_csv(os.path.join(settings['workdir'], filename, 'stats.csv'))

    def load(self, settings):
        assert os.path.join(settings['workdir']) is not None
        self.predictor = pickle.load(open(os.path.join(settings['workdir'], 'model.h5'), 'rb'))

    def build(self, settings):
        self.predictor = None

    def compile(self, settings):
        return None

    def fit(self, X, y, settings):
        self.predictor.fit(X if len(X.shape) <=2 else np.average(X, axis=1), y)

    def train(self, X, y, settings):
        assert self.predictor is not None and 'target_type' in settings and len(X.shape) == 3

        self.stats = []
        for week in np.arange(2, X.shape[1] - 1):
            logging.info('training the predictor on data till the week with id {}'.format(week))

            X_w = X[:, :week, :]

            if settings['target_type'] == 'classification':
                folds = StratifiedKFold(n_splits=settings['folds']).split(X_w, y.astype(int))
            else:
                folds = KFold(n_splits=settings['folds']).split(X_w)

            for fold, (train_index, test_index) in enumerate(folds):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                self.fit(X_train, y_train, settings)
                test_stats = self.evaluate(X_test, y_test, settings)
                self.stats.append({**{'week': week, 'fold': fold, 'y_train_idx': train_index, 'y_test_idx': test_index}, **test_stats})

        self.save(settings)
        return self.stats

    def predict(self, X, settings, proba=False):
        assert self.predictor is not None
        if proba:
            return self.predictor.predict_proba(X if len(X.shape) <= 2 else np.average(X, axis=1))
        return np.round(self.predictor.predict(X if len(X.shape) <= 2 else np.average(X, axis=1)))

    def add_grid(self, settings):
        assert self.predictor is not None
        self.predictor = GridSearchCV(self.predictor, settings['grid'], cv=settings['cv'], scoring=settings['scoring'])

    def evaluate(self, X, y, settings):
        assert 'target_type' in settings

        if settings['target_type'] == 'classification':
            return self.evaluate_classification(X, y, settings)
        else:
            return self.evaluate_regression(X, y, settings)

    def evaluate_classification(self, X, y, settings):
        y_pred = self.predict(X, settings)

        stats = {}

        stats['bal_acc'] = metrics.balanced_accuracy_score(y, y_pred)
        stats['f1'] = metrics.f1_score(y, y_pred)

        pass_ix = np.where(y==0)[0]
        fail_ix = np.where(y==1)[0]
        stats['acc_fail'] = np.sum(y_pred[fail_ix]) / len(y_pred[fail_ix])
        stats['acc_pass'] = 1 - np.sum(y_pred[pass_ix]) / len(y_pred[pass_ix])

        stats['ypred_proba'] = self.predict(X, settings, proba=True)
        stats['ypred'] = y_pred
        stats['ytrue'] = y

        fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=1)
        stats['auc'] = metrics.auc(fpr, tpr)

        return stats

    def evaluate_regression(self, X, y, settings):
        y_pred = self.predict(X, settings)
        stats = {}

        stats['mse'] = metrics.mean_squared_error(y, y_pred)

        stats['ypred_proba'] = self.predict(X, settings, proba=True)
        stats['ypred'] = y_pred
        stats['ytrue'] = y

        return stats

    def __str__(self):
        return 'Name: {}'.format(self.name)