#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.feature.feature import Feature

class ObsDurationProblem(Feature):

    def __init__(self, data, settings):
        super().__init__('obs_duration_problem' + ('_' + str(settings['ffunc']) if 'ffunc' in settings else ''))
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        if 'ffunc' in self.settings:
            self.data['prev_event'] = self.data['event_type'].shift(1)
            self.data['prev_problem_id'] = self.data['problem_id'].shift(1)
            self.data['time_diff'] = self.data['date'].diff().dt.total_seconds()
            return self.settings['ffunc'](self.data['time_diff'])
        correctProblems = self.data[(self.data['event_type'].str.contains('Problem.Check')) & (self.data['grade'] == self.settings['grade_max'])]['problem_id'].unique()
        udata = self.data[self.data['problem_id'].isin(correctProblems)]
        timeSolving = []
        for index, group in udata.groupby(by='problem_id'):
            group['time_diff'] = udata['date'].diff().dt.total_seconds()
            timeSolving += ([] if len(group.index) < 2 else [np.sum([s for s in group['time_diff'] if s >= 0 and s < 120 * 60])])
        return np.mean(timeSolving)
