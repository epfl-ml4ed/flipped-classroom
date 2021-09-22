#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import numpy as np

import logging


'''
The amount of time the student used to pass the problem, from the first tentative to the final tentative, averaged per problem
'''
class StudentSpeed(Feature):

    def __init__(self, data, settings):
        super().__init__('student_speed', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid: empty dataframe'.format(self.name))
            return Feature.INVALID_VALUE

        if not 'grade' in self.data:
            logging.debug('feature {} is invalid: no problem clickstream included'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data[self.data['event_type'].str.contains('Problem.Check') & (self.data['grade'].notnull())].sort_values(by=['problem_id', 'date'])

        if len(self.data) == 0:
            logging.debug('feature {} is invalid: no problem clickstream recorded'.format(self.name))
            return Feature.INVALID_VALUE

        self.data['prev_event'] = self.data['event_type'].shift(1)
        self.data['prev_problem_id'] = self.data['problem_id'].shift(1)
        self.data['time_diff'] = self.data['date'].diff().dt.total_seconds()
        self.data = self.data.dropna(subset=['time_diff'])
        self.data = self.data[(self.data['time_diff'] >= Feature.TIME_MIN) & (self.data['time_diff'] <= Feature.TIME_MAX)]
        self.data = self.data[self.data['problem_id'] == self.data['prev_problem_id']]

        if len(self.data) == 0:
            logging.debug('feature {} is invalid: no intervals between problem submissions present'.format(self.name))
            return Feature.INVALID_VALUE

        return np.mean(self.data.groupby(by='problem_id')['time_diff'].sum().values)
