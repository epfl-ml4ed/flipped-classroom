#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from helper.dataset.data_preparation import get_sessions, fourier_transform
from extractor.feature.feature import Feature

import numpy as np
import logging

'''
Three frequency based measures, FDH, FWH and FWD as
- FDH measures the extent to which the hourly pattern of user’s activities is repeating over days (e.g. the user is active at 8h-10h and 12h-17h on every day).
- FWH identifies if the hourly pattern of activities is repeating over weeks (e.g. in every week, the user is active at 8h-10h on Monday, 12h-17h on Tuesdays, etc.).
- FWD captures if the daily pattern of activities is repeating over weeks (e.g. the user is active on Monday and Tuesday in every week).
'''
class RegPeriodicity(Feature):

    def __init__(self, data, settings):
        super().__init__('regularity_periodicity', data, settings)

    def compute(self):
        assert 'mode' in self.settings and self.settings['timeframe'] is not 'eq-week' and self.settings['week'] > 0

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        sessions = get_sessions(self.data, self.schedule['duration'].max())
        workload = np.zeros((len(self.settings['week']), 7))
        workload[sessions['week'], sessions['weekday']] += sessions['duration']

        if self.settings['mode'] == 'm1':
            hours = self.data['date'].dt.hour.astype(int).to_list()
            n = np.arange((len(self.settings['week']) * 7 * 24 * 60 * 60) // (60 * 60))
            activity = np.array([hours.count(h) for h in np.arange(23)])
            if np.sum(activity) == 0:
                logging.info('feature {} is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            return abs(fourier_transform(activity, 1 / 24, n))

        elif self.settings['mode'] == 'm2':
            n = np.arange((len(self.settings['week']) * 7 * 24 * 60 * 60) // (60 * 60))
            self.data['hours'] = self.data['date'].dt.hour.astype(int).to_list()
            activity = np.zeros((len(self.settings['week']), 24))
            activity[self.data['week'], self.data['hours']] += 1
            return abs(fourier_transform(activity.flatten(), 1 / (7 * 24), n))

        elif self.settings['mode'] == 'm3':
            assert self.settings['timeframe'] is not 'eq-week' and self.settings['week'] > 0
            n = np.arange((len(self.settings['week']) * 7 * 24 * 60 * 60) // (24 * 60 * 60))
            weekdays = self.data['date'].dt.weekday.astype(int).to_list()
            activity = np.array([weekdays.count(h) for h in np.arange(6)])
            return abs(fourier_transform(activity, 1 / 7, n))
        else:
            raise NotImplementedError()
