#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import numpy as np
import logging

'''
The number of videos the student watched in their entirety
'''
class StudentWeeklyActiveness(Feature):

    def __init__(self, data, settings):
        super().__init__('student_weekly_activeness', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return len(self.data['Weekday'].unique()) / 7.0
