#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

from routine.compute_feature_set import main

if __name__ == "__main__":
    timeframes = ['lq_week']

    course_ids = 'mooc/progfun-005'
    #course_ids = 'mooc/EPFLx-Algebre3X-4T2015,mooc/EPFLx-Algebre3X-T1_2016,mooc/EPFLx-Algebre3X-1T2017'
    #course_ids = 'mooc/EPFLx-Algebre2X-4T2015,mooc/EPFLx-Algebre2X-T1_2016,mooc/EPFLx-Algebre2X-1T2017'
    #course_ids = 'mooc/EPFLx-AlgebreX-4T2015,mooc/EPFLx-AlgebreX-T1_2016,mooc/EPFLx-AlgebreX-1T2017'
    #course_ids = 'flipped-classroom/EPFL-CS-210-2018_t3'
    #course_ids = 'flipped-classroom/EPFL-CS-206-2019_T1'
    #course_ids = 'flipped-classroom/EPFL-AlgebreLineaire-2018,flipped-classroom/EPFL-AlgebreLineaire-2019'

    models = ['extractor.set.' + fs.split('.')[0] + '.'+ fs.split('.')[0].replace('_', '.').title().replace('.','') for fs in os.listdir('../extractor/set') if not 'pycache' in fs]

    for t in timeframes:
        for m in models:
            if 'marras' not in m:
                continue
            logging.info('{} {} {}'.format(t, course_ids, m))
            main({'model': m, 'courses': course_ids, 'timeframe': t, 'workdir': '../data/result/edm21/feature/'})