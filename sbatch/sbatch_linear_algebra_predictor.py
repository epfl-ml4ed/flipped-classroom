#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import logging
import os

from routine.train_predictor import main

logging.getLogger().setLevel(logging.INFO)

def train_individual(timeframes, course_ids, predictors, grid, base_settings):
    for timeframe in timeframes:
        for course_id in course_ids:
            feature_sets = [fs for fs in os.listdir('../data/result/edm21/feature') if timeframe in fs and course_id in fs]
            for feature_set in feature_sets:
                feature_labels = pd.read_csv(os.path.join('../data/result/edm21/feature', feature_set, 'feature_labels.csv'))
                filter_label_cols = [col for col in feature_labels.columns.tolist() if col.startswith('label-')]
                filter_label_classes = [1 for _ in filter_label_cols]
                filter_label_types = [('classification' if not 'grade' in f and not 'stopout' in f else 'regression') for f in filter_label_cols]
                for label_col, label_class, label_type in zip(filter_label_cols, filter_label_classes, filter_label_types):
                    for predictor in predictors:
                        logging.info('{} {} {} {} {}'.format(timeframe, course_id, feature_set, (label_col, label_class, label_type), predictor))
                        main({**base_settings, **{'timeframe': timeframe, 'course_id': course_id, 'model': predictor, 'target': label_col,
                                                  'target_type': label_type, 'classes': label_class, 'feature_set': feature_set,
                                                  'workdir': '../data/result/edm21', 'grid': grid[predictor.split('.')[-2]], 'cv': 5}})

def train_ensemble(timeframes, course_ids, predictors, grid, base_settings):
    for timeframe in timeframes:
        for course_id in course_ids:
            feature_sets = [fs for fs in os.listdir('../data/result/edm21/feature') if timeframe in fs and course_id in fs]
            feature_labels = pd.read_csv(os.path.join('../data/result/edm21/feature', feature_sets[0], 'feature_labels.csv'))
            filter_label_cols = [col for col in feature_labels.columns.tolist() if col.startswith('label-')]
            filter_label_classes = [1 for _ in filter_label_cols]
            filter_label_types = [('classification' if not 'grade' in f and not 'stopout' in f else 'regression') for f in filter_label_cols]
            for label_col, label_class, label_type in zip(filter_label_cols, filter_label_classes, filter_label_types):
                for predictor in predictors:
                    logging.info('{} {} {} {} {}'.format(timeframe, course_id, 'ensemble', (label_col, label_class, label_type), predictor))
                    main({**base_settings, **{'timeframe': timeframe, 'course_id': course_id, 'model': predictor, 'target': label_col, 'target_type': label_type, 'classes': label_class, 'feature_set': feature_sets, 'workdir': '../data/result/edm21'}})

if __name__ == "__main__":
    grid = {
        'random_forest': {'n_estimators': [25, 50, 100, 200], 'max_features': ('auto', 'sqrt', 'log2')},
        'svm': {'C': [1.0, 0.5], 'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'gamma': ('scale', 'auto'), 'shrinking': (True, False)},
        'dummy': {},
        'lstm_with_attention': {},
        'lstm_with_contrastive': {}
    }

    base_settings = {'batch': 512, 'folds': 10, 'hidden_units': 8, 'projection_units': 16, 'temperature': .2,
                     'epochs': 100, 'lr': .001, 'weight_init': 'he_normal', 'shuffle': True, 'verbose': 0, 'depth': 'shallow'}

    timeframes = ['eq_week', 'lq_week']
    course_ids = ['epfl_algebrelineaire']
    predictors = ['predictor.' + fs.split('.')[0] + '.' + fs.split('.')[0].replace('_', '.').title().replace('.', '') for fs in os.listdir('../predictor') if os.path.isfile(os.path.join('../predictor', fs)) and not 'pycache' in fs and not 'predictor' in fs]

    train_individual(timeframes, course_ids, predictors, grid, base_settings)

    train_ensemble(timeframes, course_ids, predictors, grid, base_settings)