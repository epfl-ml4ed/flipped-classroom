#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os

logging.getLogger().setLevel(logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from extractor.extractor_loader import ExtractorLoader
from helper.hutils import import_class

def main(settings):
    assert settings['predictor'] is not None and settings['feature_set'] is not None

    # Load feature set
    extractor = ExtractorLoader()
    extractor.load(settings)
    extractor_settings = extractor.get_settings()
    logging.info('loaded feature set {} from course {} {} {}'.format(settings['feature_set'], extractor_settings['course_id'], extractor_settings['type'], extractor_settings['platform']))

    # Arrange data
    feature_labels = extractor.get_features_values()[0][settings['target_col']].values
    feature_values = extractor.get_features_values()[1]

    X = feature_values
    y = feature_labels if settings['target_type'] == 'regression' else feature_labels.astype(int)
    logging.info('extracted data X of shape {} and y of shape {}'.format(X.shape, y.shape))

    # Initialize predictor
    predictor = import_class(settings['predictor'])()
    logging.info('initialized predictor {}'.format(settings['predictor']))

    # Start training
    logging.info('starting to train predictor {}'.format(settings['predictor']))
    predictor.train(X, y, settings)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train predictor')

    parser.add_argument('--model', dest='model', default='predictor.lstm_with_attention.LstmWithAttention', type=str, action='store')
    parser.add_argument('--target', dest='target', default='label-pass-fail', type=str, action='store')
    parser.add_argument('--target_type', dest='target_type', default='classification', type=str, action='store')
    parser.add_argument('--classes', dest='classes', default=1, type=int, action='store')
    parser.add_argument('--feature_set', dest='feature_set', default='eq_week-marras_et_al-epfl_algebrelineaire-20210131_202058', type=str, action='store')
    parser.add_argument('--depth', dest='depth', default='deep', type=str, action='store')
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