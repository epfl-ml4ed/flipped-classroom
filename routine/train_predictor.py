#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from extractor.extractor_loader import ExtractorLoader
from helper.hutils import import_class

def main(settings):
    assert settings['model'] is not None and settings['feature_set'] is not None and 'depth' in settings

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

    predictor = import_class(settings['model'])()
    if 'depth' in settings and predictor.isdepth(settings['depth']):
        logging.info('initializing {}'.format(settings['model']))

        logging.info('building {}'.format(settings['model']))
        predictor.build({**settings, **{'input_shape': X.shape[1:]}})

        logging.info('compiling {}'.format(settings['model']))
        predictor.compile(settings)

        logging.info('training and saving {}'.format(settings['model']))
        predictor.train(X, y, settings)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train predictor')

    parser.add_argument('--model', dest='model', default=None, type=str, action='store')
    parser.add_argument('--target', dest='target', default='label-pass-fail', type=str, action='store')
    parser.add_argument('--target_type', dest='target_type', default='classification', type=str, action='store')
    parser.add_argument('--classes', dest='classes', default=1, type=int, action='store')
    parser.add_argument('--feature_set', dest='feature_set', default=None, type=str, action='store')
    parser.add_argument('--workdir', dest='workdir', default='../data/result/edm21/', type=str, action='store')
    parser.add_argument('--batch', dest='batch', default=512, type=int, action='store')
    parser.add_argument('--folds', dest='folds', default=10, type=int, action='store')
    parser.add_argument('--hidden_units', dest='hidden_units', default=10, type=int, action='store')
    parser.add_argument('--epochs', dest='epochs', default=5, type=int, action='store')
    parser.add_argument('--lr', dest='lr', default=.001, type=float, action='store')
    parser.add_argument('--weight_init', dest='weight_init', default='he_normal', type=str, action='store')
    parser.add_argument('--shuffle', dest='shuffle', default=True, type=bool, action='store')
    parser.add_argument('--verbose', dest='verbose', default=1, type=int, action='store')

    settings = vars(parser.parse_args())

    main(settings)