#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

from routine.compute_feature_set import main

if __name__ == "__main__":
    timeframes = ['eq-week', 'lq-week']
    course_ids = 'flipped-classroom/EPFL-AlgebreLineaire-2018,flipped-classroom/EPFL-AlgebreLineaire-2019'
    models = ['extractor.set.' + fs.split('.')[0] + '.'+ fs.split('.')[0].replace('_', '.').title().replace('.','') for fs in os.listdir('../extractor/set') if not 'pycache' in fs]

    for t in timeframes:
        for m in models:
            logging.info('{} {} {}'.format(t, course_ids, m))
            main({'model': m, 'courses': course_ids, 'timeframe': t, 'workdir': '../data/result/edm21/feature/'})