#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

from routine.compute_feature_set import main

if __name__ == "__main__":
    timeframes = ['eq-week', 'lq-week']
    course_ids = ['flipped-classroom/EPFL-AlgebreLineaire-2018', 'flipped-classroom/EPFL-AlgebreLineaire-2019']
    models = ['extractor.set.' + fs.split('.')[0] + '.'+ fs.split('.')[0].replace('_', '.').title().replace('.','') for fs in os.listdir('../extractor/set')]

    for t in timeframes:
        for c in course_ids:
            for m in models:
                if m == 'extractor.set.__pycache__.Pycache':
                    continue
                logging.info('{} {} {}'.format(t, c, m))
                main({'model': m, 'courses': c, 'timeframe': t, 'workdir': '../data/result/edm21/feature/'})