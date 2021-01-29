#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from extractor.feature.total_clicks import TotalClicks
from extractor.feature.feature import Feature

class RatioClicksWeekendDay(Feature):

    def __init__(self, data, settings):
        super().__init__('ratio_clicks_weekend_day')
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        return TotalClicks(self.data, {**self.settings, **{'mode':'weekday'}}).compute() / (TotalClicks(self.data, {**self.settings, **{'mode':'weekend'}}).compute() + sys.float_info.epsilon)
