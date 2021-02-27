#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

from routine.compute_feature_ensemble import main

logging.getLogger().setLevel(logging.INFO)

def ensemble_creator(timeframe=None, label=None, to_remove=None):
    assert timeframe is not None and label is not None
    course_list = set([e.split('-')[2] for e in os.listdir('../data/result/edm21/feature') if not e.endswith('csv')])
    for course in course_list:
        logging.info('selecting best feature ensemble for course {}'.format(course))
        feature_list = [e for e in os.listdir('../data/result/edm21/feature') if timeframe in e and course in e and not 'ensemble' in e and (to_remove is None or to_remove is not None and not e.split('-')[1] in to_remove)]
        main({'task': 'create_features_ensemble', 'target': 'label-pass-fail', 'target_type': 'classification', 'workdir':'../data/result/edm21/', 'feature_list': feature_list, 'course': course, 'timeframe': timeframe, 'label': label})

if __name__ == "__main__":

    ensemble_creator('lq_week', 'ensemble_but_marras', ['marras_et_al'])
