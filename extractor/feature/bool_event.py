#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np

from extractor.feature.feature import Feature

'''
eq = 1 if there were events, 0 otherwise
lq = proportion of weeks with events
'''
class BoolEvent(Feature):

    def __init__(self, data, settings):
        super().__init__('text_length', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        data = self.data

        res = 0
        if 'type' in self.settings and 'week' in self.settings:
            logging.debug('filtering by event type')
            data = data[data['event_type'].str.contains(self.settings['type'].title())]
            if len(data) > 0:
                res = 1

        return res
