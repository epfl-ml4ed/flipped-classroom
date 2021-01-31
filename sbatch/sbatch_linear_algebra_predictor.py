#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import logging
import os

from routine.train_predictor import main

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    grid = {
        'mlp': {'activation': ('identity', 'logistic', 'tanh', 'relu'), 'solver': ('lbfgs', 'sgd', 'adam'), 'hidden_layer_sizes': [(8,), (16, 8), (32, 16, 8)]},
        'random_forest': {'n_estimators': [25, 50, 100, 200], 'criterion': ('gini', 'entropy'), 'max_features': ('auto', 'sqrt', 'log2')},
        'svm': {'C': [1.0, 0.5], 'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'gamma': ('scale', 'auto'), 'shrinking': (True, False)}
    }

    base_settings = {'batch': 512, 'folds': 10, 'hidden_units': 8, 'projection_units': 16, 'temperature': .2,
                     'epochs': 100, 'lr': .001, 'weight_init': 'he_normal', 'shuffle': True, 'verbose': 0,
                     'grid': grid, 'cv': 5, 'scoring': 'f1', 'depth': 'shallow'
    }

    timeframes = ['eq-week', 'lq-week']
    course_ids = ['epfl-algebrelineaire-2018', 'epfl-algebrelineaire-2019']
    predictors = ['predictor.' + fs.split('.')[0] + '.' + fs.split('.')[0].replace('_', '.').title().replace('.', '') for fs in os.listdir('../predictor') if os.path.isfile(os.path.join('../predictor', fs)) and not 'pycache' in fs and not 'predictor' in fs]

    for timeframe in timeframes:
        for course_id in course_ids:
            feature_sets = [fs for fs in os.listdir('../data/result/edm21/feature') if timeframe in fs and course_id in fs]
            for feature_set in feature_sets:
                feature_labels = pd.read_csv(os.path.join('../data/result/edm21/feature', feature_set, 'feature_labels.csv'))
                filter_label_cols = [col for col in feature_labels.columns.tolist() if col.startswith('label-')]
                filter_label_classes = [1 for f in filter_label_cols]
                filter_label_types = [('classification' if not 'grade' in f and not 'stopout' in f else 'regression') for f in filter_label_cols]
                for label_col, label_class, label_type in zip(filter_label_cols, filter_label_classes, filter_label_types):
                    for predictor in predictors:
                        logging.info('{} {} {} {} {}'.format(timeframe, course_id, feature_set, (label_col, label_class, label_type), predictor))
                        main({**base_settings, **{'timeframe': timeframe, 'course_id': course_id, 'model': predictor, 'target': label_col, 'target_type': label_type, 'classes': label_class, 'feature_set': feature_set, 'workdir': '../data/result/edm21'}})