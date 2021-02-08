#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_time_after_event
'''
The sum of time spent on a given type of events
'''
class Time(Feature):

    def __init__(self, data, settings):
        super().__init__('time_in_', data, settings)

    def compute(self):
        assert 'type' in self.settings and 'ffunc' in self.settings

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        time_intervals = get_time_after_event(self.data, self.settings['type'].title())

        if len(time_intervals) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return self.settings['ffunc'](time_intervals)
