#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from extractor.feature.feature import Feature

'''
The total number of clicks provided by a student (on a given set of days) (on a given type of events)
'''
class TextLength(Feature):

    def __init__(self, data, settings):
        super().__init__('text_length', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        data = self.data
        length = 0
        if 'type' in self.settings:
            logging.debug('filtering by event type')
            data = data[self.data['event_type'].str.contains(self.settings['type'].title())]

        if 'field' in self.settings:
            logging.debug('filtering by event type')
            length = data[self.settings['field']].str.len()

        return length
