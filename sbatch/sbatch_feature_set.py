#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from routine.compute_feature_set import main

if __name__ == "__main__":
    timeframes = ['eq_week']

    course_ids = ['flipped-classroom/EPFL-AlgebreLineaire-2018',
                  'flipped-classroom/EPFL-AlgebreLineaire-2019',
                  'flipped-classroom/EPFL-CS-206-2019_T1',
                  'flipped-classroom/EPFL-CS-210-2018_t3']

    models = ['extractor.set.extra_chen_cui.ExtraChenCui']

    for t in timeframes:
        for c in course_ids:
            for m in models:
                logging.info('{} {} {}'.format(t, course_ids, m))
                main({'model': m, 'courses': c, 'timeframe': t, 'workdir': '../data/result/lak22/feature/'})