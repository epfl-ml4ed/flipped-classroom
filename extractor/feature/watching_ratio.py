#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys

from extractor.feature.feature import Feature

from extractor.feature.attendance_rate import AttendanceRate
from extractor.feature.utilization_rate import UtilizationRate

'''
With attendance and utilization rate, the student’s overall specialty watching ratio is defined. The wathing ratio represents how student
watches the video since he/she opened it. For instance, watching_ratio=1 means that the student s completely watches the video since he/ she opened it.
'''
class WatchingRatio(Feature):

    def __init__(self, data, settings):
        super().__init__('watching_ratio', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        a = AttendanceRate(self.data, self.settings).compute()
        u = UtilizationRate(self.data, self.settings).compute()
        if a == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return a / u
