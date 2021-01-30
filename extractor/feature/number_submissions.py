#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging

from extractor.feature.feature import Feature

'''
The (statistics) on the number of problem submissions made by a student
'''
class NumberSubmissions(Feature):

    def __init__(self, data, settings):
        super().__init__('number_submissions', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data[self.data['event_type'].str.contains('Problem.Check') & (self.data['grade'].notnull())]
        self.data = self.data.merge(self.schedule, left_on='problem_id', right_on='id')

        if 'mode' in self.settings:

            if self.settings['mode'] == 'avg':
                return np.mean(self.data.groupby(by='problem_id').size().values)

            elif self.settings['mode'] == 'distinct':
                return len(self.data['problem_id'].unique())

            elif self.settings['mode'] == 'correct':
                correct = self.data[self.data['grade'] == self.data['grade_max']]['problem_id'].unique()
                return len(self.data[self.data['problem_id'].isin(correct)]) / len(correct) if len(correct) > 0 else 0

            elif self.settings['mode'] == 'perc_correct':
                return len(self.data[self.data['grade'] == self.data['grade_max']]) / len(self.data['grade'][self.data['grade'].notnull()])

            elif self.settings['mode'] == 'avg_time':
                self.data['prev_event'] = self.data['event_type'].shift(1)
                self.data['prev_problem_id'] = self.data['problem_id'].shift(1)
                self.data['time_diff'] = self.data['date'].diff().dt.total_seconds()
                self.data = self.data.dropna(subset=['time_diff'])
                self.data = self.data[(self.data['time_diff'] >= Feature.TIME_MIN) & (self.data['time_diff'] <= self.schedule['duration'].max())]
                self.data = self.data[(self.data['time_diff'] > 0.0) & (self.data['problem_id'] == self.data['prev_problem_id'])]
                return np.mean(self.data['time_diff'].values)

            elif self.settings['mode'] == 'distinct_correct':
                return len(self.data[self.data['grade'] == self.data['grade_max']]['problem_id'].unique())

        return len(self.data[self.data['grade'].notnull()])
