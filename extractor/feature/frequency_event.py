#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature
from extractor.feature.time import Time
from helper.dataset.data_preparation import count_events

import numpy as np

import logging


'''
The frequency of a given action in the clickstream.
'''
class FrequencyEvent(Feature):

    def __init__(self, data, settings):
        super().__init__('frequency_action', data, settings)

    def compute(self):
        assert 'type' in self.settings

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid: empty dataframe'.format(self.name))
            return 0.0

        if 'mode' in self.settings:
            if self.settings['mode'] == 'total':
                return count_events(self.data, self.settings['type'])

            if 'mode' in self.settings and self.settings['mode'] == 'relative':
                that_type_data = self.data[self.data['event_type'].str.contains(self.settings['type'].split('.')[0].title())]

                if len(that_type_data.index) == 0:
                    logging.debug('feature {} is invalid: no data of that type'.format(self.name))
                    return 0.0

                return count_events(self.data, self.settings['type']) / len(that_type_data)

            raise NotImplementedError()

        if self.settings['type'].lower() == 'video':
            time_settings = self.settings.copy()
            time_settings.update({'type': 'Video.Play', 'ffunc': np.sum})
            time_spent_watching = Time(self.data, time_settings).compute()
            if time_spent_watching == Feature.INVALID_VALUE or time_spent_watching == 0:
                logging.debug('feature {} is invalid: time spent watching is invalid or zero'.format(self.name))
                return Feature.INVALID_VALUE
            return count_events(self.data, self.settings['type']) / time_spent_watching

        return count_events(self.data, self.settings['type']) / len(self.data.index)