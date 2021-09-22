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
            logging.debug('feature {} is invalid: empty dataframe'.format(self.name))
            return Feature.INVALID_VALUE

        sessions = get_sessions(self.data, self.schedule['duration'].max())
        weeks = self.settings['week'] + 1
        workload = np.zeros((weeks, 7))
        workload[sessions['week'], sessions['weekday']] += sessions['duration']

        if self.settings['mode'] == 'm1':
            # Convert date to hours starting from 0
            hours = self.data['date'].values.astype(np.int64) // 10 ** 9 // 3600
            hours -= min(hours)
            period_length = weeks * 7 * 24
            activity = np.array([int(t in hours) for t in range(period_length)])  # 1 if active at hour t 0 o.w.
            if np.sum(activity) == 0:
                logging.debug('feature {} is invalid: the m1 mode is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            n = np.arange(period_length)
            return abs(fourier_transform(activity, 1 / 24, n))

        elif self.settings['mode'] == 'm2':
            period_length = weeks * 7 * 24
            hours = self.data['date'].values.astype(np.int64) // 10 ** 9 // 3600
            hours -= min(hours)
            activity = np.array([int(t in hours) for t in range(period_length)])
            n = np.arange(period_length)
            return abs(fourier_transform(activity.flatten(), 1 / (7 * 24), n))

        elif self.settings['mode'] == 'm3':
            assert self.settings['timeframe'] is not 'eq_week' and weeks > 0
            # Convert date to days starting from 0
            days = self.data['date'].values.astype(np.int64) // 10 ** 9 // (24 * 3600)
            days -= min(days)
            period_length = weeks * 7
            activity = np.array([int(d in days) for d in range(period_length)])  # 1 if active at day d 0 o.w.
            n = np.arange(period_length)
            return abs(fourier_transform(activity, 1 / 7, n))
        else:
            raise NotImplementedError()
