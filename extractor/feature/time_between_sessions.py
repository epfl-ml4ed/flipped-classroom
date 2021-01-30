#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_sessions

'''
The (statistics) on the time between sessions
'''
class TimeBetweenSessions(Feature):

    def __init__(self, data, settings):
        super().__init__('time_between_sessions', data, settings)

    def compute(self):
        assert 'ffunc' in self.settings

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        sessions = get_sessions(self.data, self.schedule['duration'].max())
        time_between_session = (sessions['end_time'] - sessions['start_time'].shift(1)).dropna().dt.total_seconds()

        if len(time_between_session) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return self.settings['ffunc'](time_between_session.values)
