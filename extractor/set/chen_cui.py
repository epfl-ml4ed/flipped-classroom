#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.extractor import Extractor

from extractor.feature.total_clicks import TotalClicks
from extractor.feature.number_sessions import NumberSessions
from extractor.feature.sum_time_sessions import SumTimeSessions
from extractor.feature.avg_time_sessions import AvgTimeSessions
from extractor.feature.std_time_sessions import StdTimeSessions
from extractor.feature.ratio_clicks_weekend_day import RatioClicksWeekendDay
from extractor.feature.sum_time import SumTime
from extractor.feature.std_time import StdTime

'''
Chen, F., & Cui, Y. (2020). Utilizing Student Time Series Behaviour in Learning Management Systems for Early Prediction of Course Performance.
In Journal of Learning Analytics, 7(2), 1-17.
'''

class ChenCui(Extractor):
    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'chen_cui'

    def extract_features(self, data, settings):
        self.features = [TotalClicks(data, settings),
                         NumberSessions(data, settings),
                         SumTimeSessions(data, settings),
                         AvgTimeSessions(data, settings),
                         StdTimeSessions(data, settings),
                         TotalClicks(data, {**settings, **{'mode':'weekday'}}),
                         TotalClicks(data, {**settings, **{'mode':'weekend'}}),
                         RatioClicksWeekendDay(data, settings),
                         TotalClicks(data, {**settings, **{'type':'problem'}}),
                         SumTime(data, {**settings, **{'mode':'problem'}}),
                         StdTime(data, {**settings, **{'mode':'problem'}})]
        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return np.nan_to_num(features)