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
        assert 'mode' in self.settings

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        sessions = get_sessions(self.data, self.schedule['duration'].max())
        weeks = self.settings['week'] + 1
        workload = np.zeros((weeks, 7))
        workload[sessions['week'], sessions['weekday']] += sessions['duration']

        if self.settings['mode'] == 'm1':
            hours = self.data['date'].dt.hour.astype(int).to_list()
            activity = np.array([hours.count(h) for h in np.arange(23)])
            if np.sum(activity) == 0:
                logging.debug('feature {} is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            return np.sum(abs(fourier_transform(activity, 1 / 24, 24)))

        elif self.settings['mode'] == 'm2':
            n = np.arange((weeks * 7 * 24 * 60 * 60) // (60 * 60))
            self.data['hours'] = self.data['date'].dt.hour.astype(int).to_list()
            activity = np.zeros((weeks, 24))
            activity[self.data['week'], self.data['hours']] += 1
            return np.sum(abs(fourier_transform(activity.flatten(), 1 / (7 * 24), 7 * 24)))

        elif self.settings['mode'] == 'm3':
            assert self.settings['timeframe'] is not 'eq_week' and weeks > 0
            weekdays = self.data['date'].dt.weekday.astype(int).to_list()
            activity = np.array([weekdays.count(h) for h in np.arange(6)])
            return np.sum(abs(fourier_transform(activity, 1 / 7, 7)))
        else:
            raise NotImplementedError()
