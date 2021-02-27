#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import pickle
import json
import time
import os

from helper.hutils import perform_scaling, perform_oversampling

class Predictor():

    def __init__(self, name=None):
        self.name = name
        self.type = 'sklearn'
        self.depth = 'shallow'
        self.hasproba = False

    def isdepth(self, depth):
        return depth == 'all' or self.depth == depth

    def exists(self, settings):
        filename = settings['timeframe'] + '-' + settings['course_id'].replace('-', '_') + '-' + settings['target_col'].replace('-', '_') + '-' + settings['predictor'].split('.')[-2] + '-' + settings['feature_set'].split('-')[1]
        return os.path.exists(os.path.join(settings['workdir'], 'predictor', filename))

    def save(self, settings):
        assert self.predictor is not None and self.predictor is not None and 'target_type' in settings

        # Set metric to show in the folder name
        filename = settings['timeframe'] + '-' + settings['course_id'].replace('-', '_') + '-' + settings['target_col'].replace('-', '_') + '-' + settings['predictor'].split('.')[-2] + '-' + settings['feature_set'].split('-')[1] + '-' + settings['avg-strategy']

        # Create the predictor main directory
        if not os.path.exists(os.path.join(settings['workdir'], 'predictor', filename)):
            os.makedirs(os.path.join(settings['workdir'], 'predictor', filename))

        # Save the trained model
        if self.type == 'sklearn':
            with open(os.path.join(settings['workdir'], 'predictor', filename, 'predictor' + '-w' + str(settings['week']) + '-f' + str(settings['fold']) + '.h5'), 'wb') as file:
                pickle.dump(self.predictor, file)
        else:
            self.predictor.save(os.path.join(settings['workdir'], 'predictor', filename, 'predictor' + '-w' + str(settings['week']) + '-f' + str(settings['fold']) + '.h5'))

        # Save train and test parameters
        with open(os.path.join(settings['workdir'], 'predictor', filename, 'params.txt'), 'w') as file:
            file.write(json.dumps(settings))

        # Save evaluation metrics
        pd.DataFrame(self.stats).to_csv(os.path.join(settings['workdir'], 'predictor', filename, 'stats.csv'), index=False)

        logging.info('saved model at {}'.format(filename))

    def load(self, settings, filename, week=None, fold=None):
        assert os.path.join(settings['workdir']) is not None and week is not None and fold is not None

        if self.type == 'sklearn':
            self.predictor = pickle.load(open(os.path.join(settings['workdir'], 'predictor', filename, 'predictor' + '-w' + str(settings['week']) + '-f' + str(settings['fold']) + '.h5'), 'rb'))
            logging.info('loaded sklearn model from pickle')
        else:
            self.predictor = tf.keras.models.load_model(os.path.join(settings['workdir'], 'predictor', filename, 'predictor' + '-w' + str(settings['week']) + '-f' + str(settings['fold']) + '.h5'))
            logging.info('loaded tensorflow model from h5')

    def build(self, settings):
        assert 'target_type' in settings
        self.predictor = None

    def prepare_data(self, X, y, settings):
        assert 'avg-strategy' in settings

        if settings['avg-strategy'] == 'none':
            Z = X
        elif settings['avg-strategy'] == 'last':
            Z = X[:, -1, :]
        elif settings['avg-strategy'] == 'avg':
            Z = np.average(X, axis=1)
        elif settings['avg-strategy'] == 'vec':
            Z = np.reshape(X, (X.shape[0], -1))
        else:
            raise NotImplementedError('the {} avg-strategy is not supported'.format(settings['avg-strategy']))

        logging.info('prepared data with X of shape {} and y of shape {} - {} from strategy {}'.format(Z.shape, y.shape if y is not None else [], [(e, list(y).count(e)) for e in np.unique(y)] if y is not None else [], settings['avg-strategy']))

        return Z, y

    def fit(self, X, y, settings):
        X, y = self.prepare_data(X, y, settings)
        self.predictor.fit(X, y)

        # Show best hyper-parameters
        if 'params_grid' in settings:
            logging.info('found best hyper-parameters {}'.format(self.predictor.best_params_))

    def train(self, X, y, settings):
        assert 'target_type' in settings and len(X.shape) == 3

        # Check whether we have already trained this combination of models
        '''
        if self.exists(settings):
            logging.info('skipped because this model combination has been already trained and saved')
            return
        '''

        # Initialize the fold creation
        folds = StratifiedKFold(n_splits=settings['folds'], shuffle=True, random_state=0).split(X, y.astype(int)) if settings['target_type'] == 'classification' else KFold(n_splits=settings['folds'], shuffle=True, random_state=0).split(X)

        # Initialize the dataframe for statistics
        self.stats = []

        # Loop for folds, then weeks
        for fold, (train_index, test_index) in enumerate(folds):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for week in np.arange(2, X.shape[1] - 1):
                tic = time.perf_counter()

                logging.info(('*' * 100))
                logging.info('working on setting with fold {} and week {}'.format(fold, week))

                X_train_w = X_train[:, :week, :]
                X_test_w = X_test[:, :week, :]

                # Impute nans as negative numbers
                X_train_w = np.nan_to_num(X_train_w, nan=-1)
                X_test_w = np.nan_to_num(X_test_w, nan=-1)

                # Build and compile the model
                Z, _ = self.prepare_data(X_test_w, None, settings)
                self.build({**settings, **{'input_shape': Z.shape[1:]}})

                # Scale the data standardly
                X_train_w, X_test_w = perform_scaling(X_train_w, X_test_w, settings['scaler'])

                # Oversample data
                X_train_w_r, y_train_r = perform_oversampling(X_train_w, y_train, settings['oversampling']) if 'oversampling' in settings else (X_train_w, y_train)

                # Fit the model for the current fold and week
                self.fit(X_train_w_r, y_train_r, settings)

                # Update the dataframe for statistics
                test_stats = self.evaluate(X_test_w, y_test, settings)
                self.stats.append({**{'week': week, 'fold': fold, 'y_train_idx': train_index, 'y_test_idx': test_index}, **test_stats})

                # Monitor overall performance based on the results already computed
                self.monitor_performance(settings)

                # Save the pre-trained model for the current fold and week
                self.save({**settings, **{'week': int(week), 'fold': int(fold)}})

                toc = time.perf_counter()
                logging.info('elapsed {:0.4f} secs for one week and fold - remaining {:0.4f} mins'.format(toc - tic, (toc - tic) * (settings['folds'] - fold) * X.shape[1] // 60))

    def predict(self, X, settings, proba=False):
        assert self.predictor is not None

        X, _ = self.prepare_data(X, None, settings)

        if proba:
            return self.predictor.predict_proba(X)[:, 1]

        return np.round(self.predictor.predict(X))

    def add_grid(self, settings):
        assert self.predictor is not None
        logging.info('added the param grid {}'.format(settings['params_grid']))
        self.predictor = GridSearchCV(self.predictor, settings['params_grid'], cv=settings['cv'], scoring='f1' if settings['target_type'] == 'classification' else 'neg_mean_squared_error')

    def evaluate(self, X, y, settings):
        assert 'target_type' in settings

        stats = self.evaluate_classification(X, y, settings) if settings['target_type'] == 'classification' else self.evaluate_regression(X, y, settings)

        metric_label = 'auc' if settings['target_type'] == 'classification' else 'rmse'
        logging.info('computed partial evaluation metrics as {}={}'.format(metric_label, stats[metric_label]))

        return stats

    def evaluate_classification(self, X, y, settings):
        stats = {}

        # Make predictions
        y_pred = self.predict(X, settings)
        y_pred_proba = self.predict(X, settings, proba=True)

        # Compute base metrics, i.e., AUC, balanced accuracy, f-measure
        fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_proba, pos_label=1)
        stats['auc'] = metrics.auc(fpr, tpr)
        stats['bal_acc'] = metrics.balanced_accuracy_score(y, y_pred)
        stats['f1'] = metrics.f1_score(y, y_pred)

        # Compute recall per class
        pass_ix = np.where(y==0)[0]
        fail_ix = np.where(y==1)[0]
        stats['acc_fail'] = np.sum(y_pred[fail_ix]) / len(y_pred[fail_ix])
        stats['acc_pass'] = 1 -np.sum(y_pred[pass_ix]) / len(y_pred[pass_ix])

        # Save fine-grained predictions
        stats['bthr'] = thresholds[np.argmax(np.sqrt(tpr * (1-fpr)))]
        stats['ypred_proba'] = y_pred_proba
        stats['ypred'] = y_pred
        stats['ytrue'] = y

        return stats

    def evaluate_regression(self, X, y, settings):
        stats = {}

        # Make predictions
        y_pred = self.predict(X, settings)

        # Compute base metrics
        stats['rmse'] = metrics.mean_squared_error(y, y_pred, squared=False)
        stats['mse'] = metrics.mean_squared_error(y, y_pred, squared=True)

        # Save fine-grained predictions
        stats['ypred'] = y_pred
        stats['ytrue'] = y

        return stats

    def monitor_performance(self, settings):
        metric_label = 'auc' if settings['target_type'] == 'classification' else 'rmse'
        metric_score = pd.DataFrame(self.stats).groupby('week').mean(metric_label)[metric_label].to_dict()
        logging.info('computed partial evaluation metrics as {}={}'.format(metric_label, metric_score))

    def __str__(self):
        return 'Name: {}'.format(self.name)