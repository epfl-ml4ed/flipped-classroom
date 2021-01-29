#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.feature.feature import Feature

class TimeSolveProblem(Feature):

    def __init__(self, data, settings):
        super().__init__('time_solve_problem')
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        timeSolving = []
        for index, group in self.data.groupby(by='problem_id'):
            timeSolving += ([] if len(group.index) < 2 else [(group['date'].tolist()[-1] - group['date'].tolist()[0]).total_seconds()])
        return np.mean(timeSolving)
