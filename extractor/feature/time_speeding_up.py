#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_time_speeding_up

'''
The amount of time spent by a student with a speed higher than 1.0 (normal)
'''
class TimeSpeedingUp(Feature):

    def __init__(self, data, settings):
        super().__init__('time_speeding_up', data, settings)

    def compute(self):
        assert 'ffunc' in self.settings and self.settings['course'].has_schedule()

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        speed_up_timings = get_time_speeding_up(self.data)
        speed_up_timings = speed_up_timings[speed_up_timings <= Feature.TIME_MAX]

        if len(speed_up_timings) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return self.settings['ffunc'](speed_up_timings)