#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.feature.feature import Feature

class NumberSubmissions(Feature):

    def __init__(self, data, settings):
        super().__init__('number_submissions' + ('_' + settings['mode'] if 'mode' in settings else '') + ('_' + settings['type'] if 'type' in settings else ''))
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data[self.data['event_type'].str.contains('Problem.Check')].index) == 0:
            return 0.0
        if 'mode' in self.settings:
            if self.settings['mode'] == 'avg_distinct':
                return np.mean(self.data[self.data['event_type'].str.contains('Problem.Check')]['problem_id'].size())
            elif self.settings['mode'] == 'distinct':
                return len(self.data[self.data['event_type'].str.contains('Problem.Check')]['problem_id'].unique())
            elif self.settings['mode'] == 'correct':
                correct = self.data[(self.data['event_type'].str.contains('Problem.Check')) & (self.data['grade'] == self.settings['grade_max'])]['problem_id'].unique()
                return len(self.data[(self.data['event_type'].str.contains('Problem.Check')) & (self.data['problem_id'].isin(correct))]) / len(correct) if len(correct) > 0 else 0
            elif self.settings['mode'] == 'perc_correct':
                return len(self.data[(self.data['event_type'].str.contains('Problem.Check')) & (self.data['grade'] == self.settings['grade_max'])]) / len(self.data['grade'][(self.data['grade']['event_type'].str.contains('Problem.Check')) & (self.data['grade']['grade'].notnull())])
            elif self.settings['mode'] == 'avg_time':
                udata = self.data[self.data['event_type'].str.contains('Problem.Check') & (self.data['grade'].notnull())]
                udata['prev_event'] = udata['event_type'].shift(1)
                udata['prev_problem_id'] = udata['problem_id'].shift(1)
                udata['time_idff'] = udata['date'].diff().dt.total_seconds()
                udata = udata[(udata['time_diff'] > 0.0) & (udata['problem_id'] == udata['prev_problem_id'])]
                return np.mean(udata['timr_diff'])
            elif self.settings['mode'] == 'distinct_correct':
                return len(self.data[(self.data['event_type'].str.contains('Problem.Check')) & (self.data['grade'] == self.settings['grade_max'])]['problem_id'].unique())
        return len(self.data[self.data['event_type'].str.contains('Problem.Check')])
