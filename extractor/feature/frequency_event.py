#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from extractor.feature.feature import Feature

from helper.dataset.data_preparation import count_events

'''
The frequency of a given action in the clickstream
'''
class FrequencyEvent(Feature):

    def __init__(self, data, settings):
        super().__init__('frequency_action', data, settings)

    def compute(self):
        assert 'type' in self.settings

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        if 'mode' in self.settings:
            if self.settings['mode'] == 'total':
                return count_events(self.data, self.settings['type'])
            if 'mode' in self.settings and self.settings['mode'] == 'relative':
                return count_events(self.data, self.settings['type']) / (self.data[self.settings['type'].split('.').lower() + '_id'].unique())
            raise NotImplementedError

        return count_events(self.data, self.settings['type']) / len(self.data.index)