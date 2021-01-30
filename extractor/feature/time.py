#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging

from extractor.feature.feature import Feature

'''
The sum of time spent on a given type of events
'''
class Time(Feature):

    def __init__(self, data, settings):
        super().__init__('time_in_', data, settings)

    def compute(self):
        assert 'type' in self.settings and 'ffunc' in self.settings

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.data['prev_event'] = self.data['event_type'].shift(1)
        self.data['prev' + self.settings['type'] + '_id'] = self.data[self.settings['type'] + '_id'].shift(1)
        self.data['time_diff'] = self.data['date'].diff().dt.total_seconds()
        self.data = self.data.dropna(subset=['time_diff'])
        self.data = self.data[(self.data['time_diff'] >= Feature.TIME_MIN) & (self.data['time_diff'] <= self.schedule['duration'].max())]

        time_intervals = self.data[(self.data['event_type'].str.contains(self.settings['type'].title())) & (self.data['prev_event'].str.contains(self.settings['type'].title()))]['time_diff'].values

        return self.settings['ffunc'](time_intervals[(time_intervals >= Feature.TIME_MIN) & (time_intervals <= self.schedule['duration'].max())])
