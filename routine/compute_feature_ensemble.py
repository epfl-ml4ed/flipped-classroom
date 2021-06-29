#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import logging
import json
import os

logging.getLogger().setLevel(logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from extractor.extractor_loader import ExtractorLoader

def main(settings):

    assert 'feature_list' in settings and settings['feature_list'] is not None

    course = settings['course']
    label = settings['label']

    # Load feature set
    logging.info('*' * 100)
    logging.info('loading feature set {}'.format(settings['feature_list']))
    extractor = ExtractorLoader()
    extractor.load(settings)

    extractor_settings = extractor.get_settings()
    extractor.save(course, {**settings, **{'feature_names': extractor_settings['feature_names'], 'course_id': extractor_settings['course_id'], 'type': extractor_settings['type'], 'platform': extractor_settings['platform']}}, label)


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