#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np
from extractor.feature.feature import Feature
from extractor.feature.time import Time
from helper.dataset.data_preparation import count_events

'''
The frequency of a given action in the clickstream.
Note: since we don't have access directly to the time spent by a user watching videos we have to infer it.
The current model measures the time between every Video.Play action and the following action.
'''
class FrequencyEvent(Feature):

    def __init__(self, data, settings):
        super().__init__('frequency_action', data, settings)

    def compute(self):
        assert 'type' in self.settings

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        if 'mode' in self.settings:
            if self.settings['mode'] == 'total':
                return count_events(self.data, self.settings['type'])
            if 'mode' in self.settings and self.settings['mode'] == 'relative':
                return count_events(self.data, self.settings['type']) / len(self.data[self.settings['type'].split('.')[0].lower() + '_id'].unique())
            raise NotImplementedError()

        if self.settings['type'].lower() == 'video':
            time_settings = self.settings.copy()
            time_settings.update({'type': 'Video.Play', 'ffunc': np.sum})
            time_spent_watching = Time(self.data, time_settings).compute()
            if time_spent_watching == Feature.INVALID_VALUE:
                logging.debug('feature {} is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            return count_events(self.data, self.settings['type']) / time_spent_watching

        return count_events(self.data, self.settings['type']) / len(self.data.index)