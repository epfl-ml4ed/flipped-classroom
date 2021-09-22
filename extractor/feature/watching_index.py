#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

from extractor.feature.attendance_rate import AttendanceRate
from extractor.feature.utilization_rate import UtilizationRate

import logging


'''
The watching index is defined as utilization rate times attendance rate
'''
class WatchingIndex(Feature):

    def __init__(self, data, settings):
        super().__init__('watching_index', data, settings)

    def compute(self):
        u = UtilizationRate(self.data, self.settings).compute()

        if u == Feature.INVALID_VALUE:
            logging.debug('feature {} is invalid: utilization rate is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        a = AttendanceRate(self.data, self.settings).compute()
        if a == Feature.INVALID_VALUE:
            logging.debug('feature {} is invalid: attendance rate is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return u * a
