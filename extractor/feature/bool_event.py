#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import logging


'''
eq = 1 if there were events, 0 otherwise
lq = proportion of weeks with events
'''
class BoolEvent(Feature):

    def __init__(self, data, settings):
        super().__init__('bool_event', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid: empty dataframe'.format(self.name))
            return 0.0

        data = self.data

        res = 0
        if 'type' in self.settings and 'week' in self.settings:
            data = data[data['event_type'].str.contains(self.settings['type'].title())]
            if len(data) > 0:
                res = 1

        return res