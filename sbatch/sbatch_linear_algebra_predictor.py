#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import logging
import os

from routine.train_predictor import main

if __name__ == "__main__":
    timeframes = ['eq-week', 'lq-week']
    course_ids = ['epfl-algebrelineaire-2018', 'epfl-algebrelineaire-2019']
    predictors = ['predictor.' + fs.split('.')[0] + '.' + fs.split('.')[0].replace('_', '.').title().replace('.', '') for fs in os.listdir('../predictor') if os.path.isfile(os.path.join('../predictor', fs)) and not 'pycache' in fs]

    for timeframe in timeframes:
        for course_id in course_ids:
            feature_sets = [os.path.join('../data/result/edm21/feature', fs) for fs in os.listdir('../data/result/edm21/feature') if timeframe in fs and course_id in fs]
            for feature_set in feature_sets:
                feature_labels = pd.read_csv(os.path.join(feature_set, 'feature_labels.csv'))
                filter_label_cols = [col for col in feature_labels.columns.tolist() if col.startswith('label-')]
                filter_label_classes = [(len(feature_labels[f].unique()) if not 'grade' in f and len(feature_labels[f].unique()) > 1 else 1) for f in filter_label_cols]
                filter_label_types = [('classification' if not 'grade' in f else 'regression') for f in filter_label_cols]
                for label_col, label_class, label_type in zip(filter_label_cols, filter_label_classes, filter_label_types):
                    for predictor in predictors:
                        logging.info('{} {} {} {} {}'.format(timeframe, course_id, feature_set, (label_col, label_class, label_type), predictor))
                        main({'model': predictor, 'target': label_col, 'target_type': label_type, 'classes': label_class, 'feature_set': feature_set, 'workdir': '../data/result/edm21'})