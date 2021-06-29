#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import logging

from helper.hcourse import init_courses
from helper.hutils import import_class

logging.getLogger().setLevel(logging.INFO)

def main(settings):

    # Load course
    courses_lst = settings['courses'].split(',')
    courses_types = list(set([c.split('/')[0] for c in courses_lst]))
    courses_ids = list(set([c.split('/')[1] for c in courses_lst]))
    courses = init_courses({'types': courses_types, 'course_ids': courses_ids, 'load': True, 'label': True})

    if len(courses) <= 0:
        raise FileNotFoundError('the courses {} do not exist'.format(settings['courses']))
    elif len(courses) == 1:
        course = courses[0]
        logging.info('only one course have been retrieved - {}'.format(settings['courses']))
    else:
        course = courses[0]
        for c in courses[1:]:
            course = course + c
        logging.info('merged multiple courses - {}'.format(settings['courses']))

    logging.info('feature extraction for {}'.format(course.course_id))
    # Initialize extractor
    extractor = import_class(settings['model'])()
    # Extract features
    extractor.extract_features_bunch(course, settings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract feature')

    parser.add_argument('--model', dest='model', default='extractor.set.akpinar_et_al.AkpinarEtAl', type=str, action='store')
    parser.add_argument('--courses', dest='courses', default='flipped-classroom/toy-course', type=str, action='store')
    parser.add_argument('--timeframe', dest='timeframe', default='eq_week', type=str, action='store')
    parser.add_argument('--workdir', dest='workdir', default='../data/result/toy/feature/', type=str, action='store')

    settings = vars(parser.parse_args())

    main(settings)