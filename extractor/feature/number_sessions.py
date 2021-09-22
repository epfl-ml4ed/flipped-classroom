#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_sessions

import logging


'''
The number of different online sessions the student has taken in the period of interest
'''
class NumberSessions(Feature):

    def __init__(self, data, settings):
        super().__init__('number_sessions', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid: empty dataframe'.format(self.name))
            return 0.0

        sessions = get_sessions(self.data, self.schedule['duration'].max())
        return len(sessions.index)
