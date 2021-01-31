#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

from scipy import stats
import numpy as np
import logging

'''
Two measures, PDH and PWD, based on the entropy of the histogram of user’s activitiy over time.
- PDH identifies if user’s activities are concentrated around a particular hour of the day.
- PWD determines if activities are concentrated around a particular day of the week.
'''
class RegPeakTime(Feature):

    def __init__(self, data, settings):
        super().__init__('regularity_peak', data, settings)

    def compute(self):
        assert 'mode' in self.settings

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        if self.settings['mode'] == 'dayhour':
            hours = self.data['date'].dt.hour.astype(int).to_list()
            activity = np.array([hours.count(h) for h in np.arange(23)])
            if np.sum(activity) == 0:
                logging.debug('feature {} is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            entropy = stats.entropy(activity / np.sum(activity))
            return (np.log2(24) - entropy) * np.max(activity)
        elif self.settings['mode'] == 'weekday':
            assert self.settings['timeframe'] is not 'eq_week' and self.settings['week'] > 0
            weekdays = self.data['date'].dt.weekday.astype(int).to_list()
            activity = np.array([weekdays.count(h) for h in np.arange(6)])
            if np.sum(activity) == 0:
                logging.debug('feature {} is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            entropy = stats.entropy(activity / np.sum(activity))
            return (np.log2(7) - entropy) * np.max(activity)
        else:
            raise NotImplementedError()

