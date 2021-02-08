#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_time_after_event

from helper.dataset.data_preparation import count_events

'''
The (statistics) on the pause lenghts
'''
class PauseDuration(Feature):

    def __init__(self, data, settings):
        super().__init__('pause_duration', data, settings)

    def compute(self):
        assert 'ffunc' in self.settings

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        pause_durations = get_time_after_event(self.data, 'Video.Pause')

        if len(pause_durations) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return self.settings['ffunc'](pause_durations)
