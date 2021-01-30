#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_sessions

'''
The (statistics) time duration of online sessions
'''
class TimeSessions(Feature):

    def __init__(self, data, settings):
        super().__init__('time_sessions', data, settings)

    def compute(self):
        assert 'ffunc' in self.settings

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        sessions = get_sessions(self.data, self.schedule['duration'].max())

        if 'mode' in self.settings:
            if self.settings['mode'] == 'length':
                return len(sessions.index)

        return self.settings['ffunc'](sessions['duration'].values)
