#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_sessions

import logging


'''
The (statistics) time duration of online sessions
'''
class TimeSessions(Feature):

    def __init__(self, data, settings):
        super().__init__('time_sessions', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid: empty dataframe'.format(self.name))
            return 0.0

        sessions = get_sessions(self.data, self.schedule['duration'].max())

        if 'mode' in self.settings:
            if self.settings['mode'] == 'length':
                return len(sessions.index)
            raise NotImplementedError()

        if len(sessions) == 0:
            logging.debug('feature {} is invalid: no sessions done by the student'.format(self.name))
            return 0.0

        durations = sessions['duration'].values

        return self.settings['ffunc'](durations)
