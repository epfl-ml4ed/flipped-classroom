#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.extractor import Extractor

from extractor.feature.number_submissions import NumberSubmissions
from extractor.feature.sum_time import SumTime
from extractor.feature.sum_time_sessions import SumTimeSessions
from extractor.feature.obs_duration_problem import ObsDurationProblem
from extractor.feature.time_solve_problem import TimeSolveProblem

'''
Wan, H., Liu, K., Yu, Q., & Gao, X. (2019). Pedagogical Intervention Practices: Improving Learning Engagement Based on Early Prediction.
In IEEE Transactions on Learning Technologies, 12(2), 278-289.
'''

class WanEtAl(Extractor):

    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'wan_et_al'

    def extract_features(self, data, settings):
        self.features = [SumTimeSessions(data.copy(), settings),
                         NumberSubmissions(data.copy(), settings),
                         NumberSubmissions(data.copy(), {**settings, **{'mode':'distinct'}}),
                         NumberSubmissions(data.copy(), {**settings, **{'mode':'distinct_correct'}}),
                         NumberSubmissions(data.copy(), {**settings, **{'mode':'avg_distinct'}}),
                         NumberSubmissions(data.copy(), {**settings, **{'mode':'correct'}}),
                         NumberSubmissions(data.copy(), {**settings, **{'mode':'perc_correct'}}),
                         SumTime(data.copy(), {**settings, **{'mode':'video'}}),
                         SumTimeSessions(data.copy(), {**settings, **{'mode':'length'}}),
                         ObsDurationProblem(data.copy(), settings),
                         TimeSolveProblem(data.copy(), settings),
                         ObsDurationProblem(data.copy(), {**settings, **{'ffunc':np.var}}),
                         ObsDurationProblem(data.copy(), {**settings, **{'ffunc':np.max}}),
                         NumberSubmissions(data.copy(), {**settings, **{'mode':'avg_time'}})]
        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return np.nan_to_num(features)
