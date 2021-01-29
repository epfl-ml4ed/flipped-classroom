#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from extractor.extractor import Extractor
from helper.hutils import import_class
from course.course import Course

def main():

    parser = argparse.ArgumentParser(description='Train autoencoder')

    parser.add_argument('--model', dest='model', default='extractor.autoencoder.vae_conv2lstm_1D.Conv2LSTM1D', type=str, action='store')
    parser.add_argument('--workdir', dest='workdir', default='../data/result/edm21/', type=str, action='store')
    parser.add_argument('--feature_set', dest='feature_set', default='20210129_024743-lalle_conati-EPFL-AlgebreLineaire-2019', type=str, action='store')
    parser.add_argument('--batch', dest='batch', default=512, type=int, action='store')
    parser.add_argument('--holdout', dest='holdout', default=1000, type=int, action='store')
    parser.add_argument('--epochs', dest='epochs', default=5, type=int, action='store')
    parser.add_argument('--latent_dim', dest='latent_dim', default=8, type=int, action='store')
    parser.add_argument('--inner_dim', dest='inner_dim', default=8, type=int, action='store')
    parser.add_argument('--layers', dest='layers', default=1, type=int, action='store')
    parser.add_argument('--dense', dest='dense', default=3, type=int, action='store')
    parser.add_argument('--lr_decay', dest='lr_decay', default=.5, type=float, action='store')
    parser.add_argument('--lr', dest='lr', default=.001, type=float, action='store')
    parser.add_argument('--beta', dest='beta', default=.5, type=float, action='store')
    parser.add_argument('--beta_1', dest='beta_1', default=.9, type=float, action='store')
    parser.add_argument('--beta_2', dest='beta_2', default=.999, type=float, action='store')
    parser.add_argument('--epsilon', dest='epsilon', default=1e-08, type=float, action='store')
    parser.add_argument('--activation', dest='activation', default='relu', type=str, action='store')
    parser.add_argument('--dropout', dest='dropout', default=.25, type=float, action='store')
    parser.add_argument('--verbose', dest='verbose', default=1, type=int, action='store')
    parser.add_argument('--shuffle', dest='shuffle', default=True, type=bool, action='store')
    parser.add_argument('--weight_init', dest='weight_init', default='he_normal', type=str, action='store')
    parser.add_argument('--lr_reduce', dest='lr_reduce', default=False, type=bool, action='store')
    parser.add_argument('--conv_filter_n', dest='conv_filter_n', default=64, type=int, action='store')

    settings = vars(parser.parse_args())

    # Load feature set
    logging.info('loading feature set {}'.format(settings['feature_set']))
    extractor = Extractor()
    extractor.load(settings)

    # Load associated course
    logging.info('loading course data from {}'.format(settings['feature_set']))
    extractor_settings = extractor.get_settings()
    course = Course(extractor_settings['course_id'], extractor_settings['type'], extractor_settings['platform'])
    course.load()

    # Arrange data
    logging.info('arranging data from {}'.format(course.course_id))
    X = extractor.get_features_values()[1]

    users, timesteps, obs_per_timestep = X.shape
    features = timesteps * obs_per_timestep

    logging.info('initializing model {}'.format(settings['model']))
    ae = import_class(settings['model'])()

    logging.info('building model {}'.format(settings['model']))
    ae.build(features=features, obs_per_timestep=obs_per_timestep, settings=settings)

    logging.info('compiling model {}'.format(settings['model']))
    ae.compile(settings=settings)

    logging.info('training model {}'.format(settings['model']))
    ae.train(X=np.reshape(X, (users, -1)), settings=settings)

if __name__ == "__main__":
    main()