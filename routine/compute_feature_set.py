#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse

from extractor.set.lalle_conati import LalleConati
from helper.hcourse import init_courses
from helper.hutils import import_class

def main():

    parser = argparse.ArgumentParser(description='Extract feature')

    parser.add_argument('--model', dest='model', default='extractor.set.lalle_conati.LalleConati', type=str, action='store')
    parser.add_argument('--workdir', dest='workdir', default='../data/result/edm21/', type=str, action='store')
    parser.add_argument('--course', dest='course', default='flipped-classroom/EPFL-AlgebreLineaire-2019', type=str, action='store')
    parser.add_argument('--timeframe', dest='timeframe', default='eq-week', type=str, action='store')
    parser.add_argument('--max_session_length', dest='max_session_length', default=120, type=int, action='store')
    parser.add_argument('--min_actions', dest='min_actions', default=10, type=int, action='store')
    parser.add_argument('--filepath', dest='filepath', default='../data/result/edm21/feature/', type=str, action='store')
    parser.add_argument('--grade_max', dest='grade_max', default=100.0, type=float, action='store')

    settings = vars(parser.parse_args())

    # Load course
    course = init_courses({'types': [settings['course'].split('/')[0]], 'course_ids': [settings['course'].split('/')[1]], 'load': True, 'label': True})[0]

    # Initialize extractor
    extractor = import_class(settings['model'])()

    # Extract features
    extractor.load_features(course, settings)

    # Show final feature shape
    print('feature shape', extractor.get_features_values()[1].shape)

if __name__ == "__main__":
    main()