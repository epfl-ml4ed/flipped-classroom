#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.extractor import Extractor

from extractor.feature.number_submissions import NumberSubmissions
from extractor.feature.time import Time
from extractor.feature.time_sessions import TimeSessions
from extractor.feature.obs_duration_problem import ObsDurationProblem
from extractor.feature.time_solve_problem import TimeSolveProblem

'''
Wan, H., Liu, K., Yu, Q., & Gao, X. (2019). Pedagogical Intervention Practices: Improving Learning Engagement Based on Early Prediction.
In IEEE Transactions on Learning Technologies, 12(2), 278-289.

The following features cannot be computed, based on the data we currently have:
- number_forum_posts
- number_forum_browse
- number forum responses
- average number of submissions percentile
- average number of submissions percent
'''

class WanEtAl(Extractor):

    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'wan_et_al'

    def extract_features(self, data, settings):
        self.features = [TimeSessions(data, {**settings, **{'ffunc': np.sum}}),
                         NumberSubmissions(data, {**settings, **{'mode': 'distinct'}}),
                         NumberSubmissions(data, settings),
                         NumberSubmissions(data, {**settings, **{'mode': 'distinct_correct'}}),
                         NumberSubmissions(data, {**settings, **{'mode': 'avg'}}),
                         NumberSubmissions(data, {**settings, **{'mode': 'avg_time'}}),
                         ObsDurationProblem(data, settings),
                         NumberSubmissions(data, {**settings, **{'mode': 'perc_correct'}}),
                         TimeSolveProblem(data, settings),
                         ObsDurationProblem(data, {**settings, **{'ffunc': np.var}}),
                         ObsDurationProblem(data, {**settings, **{'ffunc': np.max}}),
                         Time(data, {**settings, **{'mode': 'video'}}),
                         TimeSessions(data, {**settings, **{'mode': 'length'}}),
                         NumberSubmissions(data, {**settings, **{'mode': 'correct'}})]

        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return features

