#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from extractor.feature.feature import Feature

from extractor.feature.attendance_rate import AttendanceRate
from extractor.feature.utilization_rate import UtilizationRate

class WatchingIndex(Feature):

    def __init__(self, data, settings):
        super().__init__('watching_index')
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        return UtilizationRate(self.data, self.settings).compute() * AttendanceRate(self.data, self.settings).compute()
