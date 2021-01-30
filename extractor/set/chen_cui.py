#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.extractor import Extractor

from extractor.feature.total_clicks import TotalClicks
from extractor.feature.number_sessions import NumberSessions
from extractor.feature.time_sessions import TimeSessions
from extractor.feature.time_between_sessions import TimeBetweenSessions
from extractor.feature.ratio_clicks_weekend_day import RatioClicksWeekendDay
from extractor.feature.time import Time

'''
Chen, F., & Cui, Y. (2020). Utilizing Student Time Series Behaviour in Learning Management Systems for Early Prediction of Course Performance.
In Journal of Learning Analytics, 7(2), 1-17.

The following features cannot be computed, based on the data we currently have:
- Number of clicks on campus
- Ratio of on-campus to off-campus clicks
- Number of clicks for module “Forum”
- Number of clicks for module “Overview report”
- Number of clicks for module “System”
- Number of clicks for module “User report”
'''

class ChenCui(Extractor):
    def __init__(self, name='base'):
        super().__init__(name)
        self.name = 'chen_cui'

    def extract_features(self, data, settings):
        self.features = [TotalClicks(data, settings),
                         NumberSessions(data, settings),
                         TimeSessions(data, {**settings, **{'ffunc': np.sum}}),
                         TimeSessions(data, {**settings, **{'ffunc': np.mean}}),
                         TimeBetweenSessions(data, {**settings, **{'ffunc': np.std}}),
                         TimeSessions(data, {**settings, **{'ffunc': np.std}}),
                         TotalClicks(data, {**settings, **{'mode':  'weekday'}}),
                         TotalClicks(data, {**settings, **{'mode':'weekend'}}),
                         RatioClicksWeekendDay(data, settings),
                         TotalClicks(data, {**settings, **{'type':' video'}}),
                         TotalClicks(data, {**settings, **{'type':' problem'}}),
                         Time(data, {**settings, **{'type': 'problem', 'ffunc': np.sum}}),
                         Time(data, {**settings, **{'type': 'video', 'ffunc': np.sum}})]
        features = [f.compute() for f in self.features]
        assert len(features) == self.__len__()
        return features