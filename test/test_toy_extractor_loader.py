#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging

from extractor.extractor_loader import ExtractorLoader
from helper.hutils import import_class

logging.getLogger().setLevel(logging.INFO)

def main(settings):
    assert settings['feature_set'] is not None

    # Load feature set
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

    X = feature_values
    y = feature_labels if settings['target_type'] == 'regression' else feature_labels.astype(int)

    logging.info(X.shape)
    logging.info(y.shape)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load features')

    parser.add_argument('--feature_set', dest='feature_set', default='eq_week-akpinar_et_al-toy_course_20210202_000840-20210202_112417', type=str, action='store') # Find the folder in data/result/test/feature
    parser.add_argument('--target', dest='target', default='label-pass-fail', type=str, action='store')
    parser.add_argument('--target_type', dest='target_type', default='classification', type=str, action='store')
    parser.add_argument('--classes', dest='classes', default=1, type=int, action='store')
    parser.add_argument('--workdir', dest='workdir', default='../data/result/test/', type=str, action='store')

    settings = vars(parser.parse_args())

    main(settings)