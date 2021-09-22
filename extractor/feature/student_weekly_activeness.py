#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import logging


'''
The number of videos the student watched in their entirety
'''
class StudentWeeklyActiveness(Feature):

    def __init__(self, data, settings):
        super().__init__('student_weekly_activeness', data, settings)

    def compute(self):

        return len(self.data['weekday'].unique()) / 7.0
