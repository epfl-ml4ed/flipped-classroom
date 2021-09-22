#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_time_after_event

import logging


'''
The sum of time spent on a given type of events
'''
class Time(Feature):

    def __init__(self, data, settings):
        super().__init__('time_in_', data, settings)

    def compute(self):
        assert 'type' in self.settings and 'ffunc' in self.settings

        if '.' in self.settings['type']:
            time_after_event = get_time_after_event(self.data, self.settings['type'])

            if len(time_after_event) == 0:
                logging.debug('feature {} is invalid: no time after an event present'.format(self.name))
                return 0.0

            return self.settings['ffunc'](time_after_event)

        if not self.settings['type'] + '_id' in self.data:
            logging.debug('feature {} is invalid: no video nor problem type inserted'.format(self.name))
            return 0.0

        self.data['prev_event'] = self.data['event_type'].shift(1)
        self.data['prev_' + self.settings['type'] + '_id'] = self.data[self.settings['type'] + '_id'].shift(1)
        self.data['time_diff'] = self.data['date'].diff().dt.total_seconds()
        self.data = self.data.dropna(subset=['time_diff'])
        self.data = self.data[(self.data['time_diff'] >= Feature.TIME_MIN) & (self.data['time_diff'] <= Feature.TIME_MAX)]

        time_intervals = self.data[(self.data['prev_event'].str.contains(self.settings['type'].title()))]['time_diff'].values
        time_intervals = time_intervals[(time_intervals >= Feature.TIME_MIN) & (time_intervals <= Feature.TIME_MAX)]

        if len(time_intervals) == 0:
            logging.debug('feature {} is invalid: no time intervals computable'.format(self.name))
            return 0.0

        return self.settings['ffunc'](time_intervals)
