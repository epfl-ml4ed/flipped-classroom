#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

from extractor.feature.attendance_rate import AttendanceRate
from extractor.feature.utilization_rate import UtilizationRate

import logging


'''
With attendance and utilization rate, the student’s overall specialty watching ratio is defined. The wathing ratio represents how student
watches the video since he/she opened it. For instance, watching_ratio=1 means that the student s completely watches the video since he/ she opened it.
'''
class WatchingRatio(Feature):

    def __init__(self, data, settings):
        super().__init__('watching_ratio', data, settings)

    def compute(self):
        u = UtilizationRate(self.data, self.settings).compute()
        if u == Feature.INVALID_VALUE or u == 0:
            logging.debug('feature {} is invalid: utilization rate is invalid or zero'.format(self.name))
            return Feature.INVALID_VALUE

        a = AttendanceRate(self.data, self.settings).compute()
        if a == Feature.INVALID_VALUE:
            logging.debug('feature {} is invalid: attendance rate is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return a / u
