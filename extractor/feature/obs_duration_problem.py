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
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        if 'ffunc' in self.settings:
            self.data['prev_event'] = self.data['event_type'].shift(1)
            self.data['prev_problem_id'] = self.data['problem_id'].shift(1)
            self.data['time_diff'] = self.data['date'].diff().dt.total_seconds()
            self.data = self.data.dropna(subset=['time_diff'])
            self.data = self.data[(self.data['time_diff'] >= Feature.TIME_MIN) & (self.data['time_diff'] <= self.schedule['duration'].max())]
            return self.settings['ffunc'](self.data['time_diff'].values)

        return TimeSessions(self.data, {**self.settings, **{'ffunc': np.sum}}).compute() / NumberSubmissions(self.data, {**self.settings, **{'mode': 'distinct_correct'}}).compute()
