#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from extractor.feature.feature import Feature

from extractor.feature.attendance_rate import AttendanceRate
from extractor.feature.utilization_rate import UtilizationRate

'''
The watching index is defined as utilization rate times attendance rate
'''
class WatchingIndex(Feature):

    def __init__(self, data, settings):
        super().__init__('watching_index', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return UtilizationRate(self.data, self.settings).compute() * AttendanceRate(self.data, self.settings).compute()
