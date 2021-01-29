#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from extractor.extractor import Extractor
from helper.hutils import import_class
from course.course import Course

def main():

    parser = argparse.ArgumentParser(description='Train predictor')

    parser.add_argument('--model', dest='model', default='predictor.random_forest.RandomForest', type=str, action='store')
    parser.add_argument('--workdir', dest='workdir', default='../data/result/edm21/', type=str, action='store')
    parser.add_argument('--target', dest='target', default='pass-fail', type=str, action='store')
    parser.add_argument('--feature_set', dest='feature_set', default='20210129_024743-lalle_conati-EPFL-AlgebreLineaire-2019', type=str, action='store')
    parser.add_argument('--batch', dest='batch', default=512, type=int, action='store')
    parser.add_argument('--folds', dest='folds', default=5, type=int, action='store')
    parser.add_argument('--hidden_units', dest='hidden_units', default=10, type=int, action='store')
    parser.add_argument('--epochs', dest='epochs', default=5, type=int, action='store')
    parser.add_argument('--classes', dest='classes', default=1, type=int, action='store')
    parser.add_argument('--lr', dest='lr', default=.001, type=float, action='store')
    parser.add_argument('--activation', dest='activation', default='sigmoid', type=str, action='store')
    parser.add_argument('--verbose', dest='verbose', default=1, type=int, action='store')
    parser.add_argument('--shuffle', dest='shuffle', default=True, type=bool, action='store')
    parser.add_argument('--weight_init', dest='weight_init', default='he_normal', type=str, action='store')
    parser.add_argument('--metrics', dest='metrics', default='accuracy', type=str, action='store')
    parser.add_argument('--loss', dest='loss', default='binary_crossentropy', type=str, action='store')

    settings = vars(parser.parse_args())

    assert settings['model'] is not None and settings['feature_set'] is not None

    # Load feature set
    logging.info('> loading feature set {}'.format(settings['feature_set']))
    extractor = Extractor()
    extractor.load(settings)

    # Load associated course
    logging.info('> loading course data from {}'.format(settings['feature_set']))
    extractor_settings = extractor.get_settings()
    course = Course(extractor_settings['course_id'], extractor_settings['type'], extractor_settings['platform'])
    course.load()
    course.label()

    # Arrange data
    logging.info('> arranging data from {}'.format(course.course_id))
    U = extractor.get_features_values()[0]
    X = extractor.get_features_values()[1]
    y = course.get_clickstream_grade().set_index('user_id').reindex(U)[settings['target']].values

    logging.info('> initializing {}'.format(settings['model']))
    predictor = import_class(settings['model'])()

    logging.info('> building {}'.format(settings['model']))
    predictor.build({**settings, **{'input_shape': X.shape[1:]}})

    logging.info('> compiling {}'.format(settings['model']))
    predictor.compile({**settings, **{'metrics': settings['metrics'].split(',')}})

    logging.info('> training and saving {}'.format(settings['model']))
    predictor.train(X, y, settings)

if __name__ == "__main__":
    main()