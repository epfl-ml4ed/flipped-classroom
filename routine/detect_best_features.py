#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import numpy as np
import argparse
import logging
import json
import os

logging.getLogger().setLevel(logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from extractor.extractor_loader import ExtractorLoader

def main(settings):

    assert 'feature_set' in settings and settings['feature_set'] is not None

    # Check existence
    filepath = os.path.join(settings['workdir'], 'feature', settings['feature_set'], 'feature_selected.txt')
    if os.path.exists(filepath):
        logging.info('features for {} already selected at {}'.format(settings['feature_set'], filepath))
        return

    # Load feature set
    logging.info('*' * 100)
    logging.info('loading feature set {}'.format(settings['feature_set']))
    extractor = ExtractorLoader()
    extractor.load(settings)

    # Load associated course
    extractor_settings = extractor.get_settings()
    logging.info('loading course data from {} {} {}'.format(extractor_settings['course_id'], extractor_settings['type'], extractor_settings['platform']))

    # Arrange data
    logging.info('arranging data from {}'.format(extractor_settings['course_id']))
    feature_labels = extractor.get_features_values()[0][settings['target']].values
    feature_values = extractor.get_features_values()[1]
    feature_settings = extractor.get_settings()

    assert len(feature_settings['feature_names']) == feature_values.shape[2]

    X = feature_values
    y = feature_labels if settings['target_type'] == 'regression' else feature_labels.astype(int)

    logging.info('computing best features for {}'.format(settings['feature_set']))

    # Prepare data
    X = np.nan_to_num(X)
    logging.info('working on X with shape {}'.format(X.shape))

    X_avg = np.average(X, axis=1)
    logging.info('taken the average vector across weeks')

    logging.info('averaging on X_avg with shape {}'.format(X_avg.shape))

    # Estimate best features
    param_grid = {'n_estimators': [10, 25, 50, 100, 200, 500], 'max_features': ['sqrt', 'log2', None]}

    estimator = model_selection.GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=0), param_grid=param_grid, cv=10, scoring='roc_auc')
    estimator.fit(X_avg, y)
    logging.info('found best score {}'.format(estimator.best_score_))

    selector = SelectFromModel(estimator.best_estimator_, threshold=settings['mean_weight']+'*mean')
    selector.fit(X_avg, y)

    # Update settings
    settings['best_score_before'] = estimator.best_score_
    settings['support'] = list(selector.get_support().astype(float))
    settings['importance'] = list(selector.estimator_.feature_importances_)
    settings['feature_names'] = feature_settings['feature_names']

    assert len(settings['support']) == len(settings['importance'])

    logging.info('found n={} with ranking r={}'.format(np.sum(settings['support']), [(enum, settings['importance'][idx], np.array(settings['feature_names'])[idx]) for enum, idx in enumerate(np.argsort(settings['importance'])[::-1])]))
    logging.info('computed importance {}'.format(settings['importance']))

    # Checking drop
    estimator = model_selection.GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=0), param_grid=param_grid, cv=10, scoring='roc_auc')
    estimator.fit(X_avg[:, np.array(settings['support']).astype(bool)], y)
    logging.info('dropped to {} best score with only the selected features'.format(estimator.best_score_))
    settings['best_score_after'] = estimator.best_score_

    # Saving selections
    with open(filepath, 'w') as file:
        file.write(json.dumps(settings))
    logging.info('best features saved in {}'.format(filepath))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Find best features')

    parser.add_argument('--task', dest='task', default='detect_best_features', type=str, action='store')
    parser.add_argument('--target', dest='target', default='label-pass-fail', type=str, action='store')
    parser.add_argument('--target_type', dest='target_type', default='classification', type=str, action='store')
    parser.add_argument('--feature_set', dest='feature_set', default='eq_week-marras_et_al-epfl_cs_206_2019_t1', type=str, action='store')
    parser.add_argument('--workdir', dest='workdir', default='../data/result/edm21/', type=str, action='store')
    parser.add_argument('--verbose', dest='verbose', default=1, type=int, action='store')

    settings = vars(parser.parse_args())

    main(settings)