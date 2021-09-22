#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from helper.dataset.data_preparation import similarity_days, chi2_divergence
from extractor.feature.feature import Feature

from scipy.spatial.distance import jensenshannon
from datetime import datetime
import numpy as np

import logging


'''
Three measures WS1, WS2 and WS3 based on the similarity between weekly profiles of user’s activities.
- WS1 measures if the user works on the same weekdays.
- WS2 compares the normalized profiles and measures if user has a similar distribution of workload among weekdays, in different weeks of the course.
- WS3 compares the original profiles and reflects if the time spent on each day of the week is similar for different weeks of the course.
'''
class RegWeeklySim(Feature):

    def __init__(self, data, settings):
        super().__init__('regularity_weekly_similarity', data, settings)

    def compute(self):
        assert 'mode' in self.settings

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid: empty dataframe'.format(self.name))
            return Feature.INVALID_VALUE

        weeks = self.settings['week'] + 1
        workload = np.zeros((weeks, 7))
        workload[self.data['week'], self.data['weekday']] += 1
        hist = workload / np.sum(workload)

        # Hours of activity starting at midnight of the first timestamp
        hours = (self.data['date']).values.astype(np.int64) // 10 ** 9 // 3600
        min_day = self.data.date.min() # First day of activity
        # Make the hours start from midnight of the first day
        hours -= int(datetime(min_day.year, min_day.month, min_day.day).timestamp() / 3600)

        period_length = weeks * 7 * 24
        activity = np.array([int(t in hours) for t in range(period_length)]).reshape((weeks, 7 * 24))
        activity = np.array([week.reshape((7, 24)).sum(axis=1) for week in activity])  # shape (weeks, 7)
        if self.settings['mode'] == 'm1':
            return np.mean([similarity_days(workload[i], workload[j]) for i in range(workload.shape[0]) for j in range(i+1, workload.shape[0])])
        elif self.settings['mode'] == 'm2':
            res = []
            for i in range(activity.shape[0]):
                for j in range(i + 1, activity.shape[0]):
                    if not activity[i].any() or not activity[j].any():
                        continue
                    res.append(1 - jensenshannon(activity[i], activity[j], 2.0))
            if len(res) == 0:
                logging.debug('feature {} is invalid: the m2 mode is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            return np.mean(np.clip(np.nan_to_num(res), 0, 1))
        elif self.settings['mode'] == 'm3':
            res = []
            for i in range(activity.shape[0]):
                for j in range(i + 1, activity.shape[0]):
                    if not activity[i].any() or not activity[j].any():
                        continue
                    res.append(chi2_divergence(activity[i], activity[j], hist[i], hist[j]))
            if len(res) == 0:
                return np.nan
            return np.mean(np.nan_to_num(res))
        else:
            raise NotImplementedError()
