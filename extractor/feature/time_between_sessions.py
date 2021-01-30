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
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        sessions = get_sessions(self.data, self.schedule['duration'].max())
        time_between_session = (sessions['end_date'] - sessions['start_date'].shift(1)).dropna().total_seconds()
        return self.settings['ffunc'](time_between_session.values)
