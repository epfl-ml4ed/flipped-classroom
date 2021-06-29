#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from extractor.feature.feature import Feature

'''
The total number of clicks provided by a student (on a given set of days) (on a given type of events)
'''
class TotalClicks(Feature):

    def __init__(self, data, settings):
        super().__init__('total_clicks', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        data = self.data

        if 'mode' in self.settings:
            if self.settings['mode'] == 'weekend':
                logging.debug('filtering by weekend')
                data = data[data['weekday'].isin(Feature.WEEKEND)]
            if self.settings['mode'] == 'weekday':
                logging.debug('filtering by weekday')
                data = data[data['weekday'].isin(Feature.WEEKDAY)]

        if 'type' in self.settings:
            logging.debug('filtering by event type')
            data = data[self.data['event_type'].str.contains(self.settings['type'].title())]

        return len(data.index)
