#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging

from extractor.feature.feature import Feature
from extractor.feature.time_sessions import TimeSessions
from extractor.feature.number_submissions import NumberSubmissions

'''
The total time needed to solve a problem, from the first to the last submission
'''
class ObsDurationProblem(Feature):

    def __init__(self, data, settings):
        super().__init__('obs_duration_problem', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        if 'ffunc' in self.settings:
            self.data['prev_problem_id'] = self.data['problem_id'].shift(1)
            self.data['time_diff'] = self.data['date'].diff().dt.total_seconds()
            self.data = self.data.dropna(subset=['time_diff'])
            self.data = self.data[(self.data['prev_problem_id'] == self.data['problem_id']) &
                                  (self.data['time_diff'] >= Feature.TIME_MIN) &
                                  (self.data['time_diff'] <= Feature.TIME_MAX)]
            time_intervals = self.data['time_diff'].values
            if len(time_intervals) == 0:
                logging.debug('feature {} is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            return self.settings['ffunc'](time_intervals)

        no_submissions = NumberSubmissions(self.data, {**self.settings, **{'mode': 'distinct_correct'}}).compute()
        no_sessions = TimeSessions(self.data, {**self.settings, **{'ffunc': np.sum}}).compute()
        if no_submissions == 0 or no_submissions == Feature.INVALID_VALUE or no_sessions == Feature.INVALID_VALUE:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return no_sessions / no_submissions
