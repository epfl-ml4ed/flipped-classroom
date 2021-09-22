#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import logging


'''
The total number of clicks provided by a student (on a given set of days) (on a given type of events)
'''
class TotalClicks(Feature):

    def __init__(self, data, settings):
        super().__init__('total_clicks', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid: empty dataframe'.format(self.name))
            return 0.0

        data = self.data

        if 'mode' in self.settings:
            if self.settings['mode'] == 'weekend':
                data = data[data['weekday'].isin(Feature.WEEKEND)]
            if self.settings['mode'] == 'weekday':
                data = data[data['weekday'].isin(Feature.WEEKDAY)]

        if 'type' in self.settings:
            data = data[self.data['event_type'].str.contains(self.settings['type'].title())]

        return len(data.index)
