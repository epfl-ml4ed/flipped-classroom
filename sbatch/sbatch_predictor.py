#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import logging
import os

from routine.train_predictor import main

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":

    # Customize parameters
    params_grid = {
        'random_forest': {'n_estimators': [25, 50, 100, 200, 300, 500], 'max_features': ['sqrt', None, 'log2'], 'criterion': ['gini', 'entropy']},
        'gradient_boosting': {'n_estimators': [5, 10, 25, 50, 100, 200, 500], 'max_features':  ['auto', 'sqrt', 'log2', None]},
        'dummy': {'strategy': ['stratified', 'most_frequent', 'prior', 'uniform']},
        'svm': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'poly', 'sigmoid'], 'degree': [3, 5, 10, 20]},
        'dnn': {'hidden_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.01, 'batch_size': 256, 'epochs': 100, 'verbose': 1, 'shuffle': True},
        'lstm': {'hidden_units': 32, 'dropout_rate': 0.2, 'learning_rate': 0.01, 'batch_size': 64, 'epochs': 40, 'verbose': 1, 'shuffle': True},
        'lstm_with_attention': {'hidden_units': 8, 'dropout_rate': 0.2, 'learning_rate': 0.01, 'batch_size': 128, 'epochs': 150, 'verbose': 1, 'shuffle': True},
        'lstm_with_contrastive': {'temperature': .1, 'projection_units': 64, 'hidden_units': 512, 'dropout_rate': 0.2, 'learning_rate': 0.01, 'batch_size': 256, 'epochs': 200, 'verbose': 1, 'shuffle': True}
    }

    config_grid = {'folds': 10, 'cv': 10, 'shuffle': True, 'verbose': 0, 'depth': 'deep', 'selected_features': True, 'avg-strategy': 'none', 'scaler': 'none', 'oversampling': 'none'}
    label_grid = {'label-pass-fail': 'classification', 'label-grade': 'regression', 'label-dropout': 'classification', 'label-stopout': 'regression'}
    workdir_feature = '../data/result/edm21/feature'

    # Retrieve all the timeframes, courses, feature sets, and predictors
    timeframes = sorted({fs.split('-')[0] for fs in os.listdir(workdir_feature) if not '.' in fs})
    course_ids = sorted({fs.split('-')[2] for fs in os.listdir(workdir_feature) if not '.' in fs})
    feature_sets = sorted({fs.split('-')[1] for fs in os.listdir(workdir_feature) if not '.' in fs})
    predictors = sorted({'predictor.' + fs.split('.')[0] + '.' + fs.split('.')[0].replace('_', '.').title().replace('.', '') for fs in os.listdir('../predictor') if os.path.isfile(os.path.join('../predictor', fs)) and not 'pycache' in fs and not 'predictor' in fs})

    logging.info('found timeframes {}'.format(timeframes))
    logging.info('found course ids {}'.format(course_ids))
    logging.info('found feature sets {}'.format(feature_sets))
    logging.info('found predictors {}'.format(predictors))

    # Filtered combinations of timeframes, courses, feature sets, and predictors
    target_label_col = 'label-pass-fail'
    timeframes = sorted(set(timeframes) & {'lq_week'})
    course_ids = sorted(set(course_ids) & {'epfl_algebrelineaire'})  #, 'epfl_cs_210_2018_t3', 'epflx_algebrex', 'progfun_005', 'epfl_algebrelineaire', 'epfl_cs_206_2019_t1'})
    feature_sets = sorted(set(feature_sets) & {'marras_et_al'})
    predictors = sorted(set(predictors) & {'predictor.lstm.Lstm'})

    logging.info('filtered timeframes {}'.format(timeframes))
    logging.info('filtered course ids {}'.format(course_ids))
    logging.info('filtered feature sets {}'.format(feature_sets))
    logging.info('filtered predictors {}'.format(predictors))

    # Loop over all the combinations
    for timeframe in timeframes:

        for course_id in course_ids:
            selected_feature_sets = [fs for fs in os.listdir(workdir_feature) if timeframe in fs and course_id in fs and fs.split('-')[1] in feature_sets]

            for feature_set in selected_feature_sets:
                feature_labels = pd.read_csv(os.path.join(workdir_feature, feature_set, 'feature_labels.csv'))
                filter_label_cols = [col for col in feature_labels.columns.tolist() if col.startswith(target_label_col)]
                filter_label_classes = [len(set(feature_labels[col])) for col in filter_label_cols]
                filter_label_types = [label_grid[col] for col in filter_label_cols]

                for label_col, label_class, label_type in zip(filter_label_cols, filter_label_classes, filter_label_types):

                    for predictor in predictors:
                        logging.info('*' * 100)
                        logging.info('timeframe={} course_id={} feature_set={} label_info={} predictor={}'.format(timeframe, course_id, feature_set, (label_col, label_class, label_type), predictor))
                        main({**config_grid, **{'workdir':'../data/result/edm21/', 'timeframe': timeframe, 'course_id': course_id, 'feature_set': feature_set, 'target_col': label_col, 'target_type': label_type, 'target_classes': label_class, 'predictor': predictor, 'params_grid': params_grid[predictor.split('.')[-2]]}})