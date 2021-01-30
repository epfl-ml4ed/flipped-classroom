#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from helper.dataset.data_preparation import get_sessions, similarity_days, chi2_divergence
from extractor.feature.feature import Feature

from scipy.spatial.distance import jensenshannon
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
        assert 'mode' in self.settings and self.settings['timeframe'] is not 'eq-week' and self.settings['week'] > 0

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        workload = np.zeros((len(self.settings['week']), 7))
        workload[self.data['week'], self.data['weekday']] += 1
        hist = workload / np.sum(workload)

        if self.settings['mode'] == 'm1':
            return np.mean([similarity_days(workload[i], workload[j]) for i in range(workload.shape[0]) for j in range(i+1, workload.shape[0])])
        elif self.settings['mode'] == 'm2':
            res = []
            for i in range(workload.shape[0]):
                for j in range(i + 1, workload.shape[0]):
                    if not workload[i].any() or not workload[j].any():
                        continue
                    res.append(1 - jensenshannon(workload[i], workload[j], 2.0))
            if len(res) == 0:
                logging.info('feature {} is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            return np.mean(np.clip(np.nan_to_num(res), 0, 1))
        elif self.settings['mode'] == 'm3':
            res = []
            for i in range(workload.shape[0]):
                for j in range(i + 1, workload.shape[0]):
                    if not workload[i].any() or not workload[j].any():
                        continue
                    res.append(chi2_divergence(workload[i], workload[j], hist[i], hist[j]))
            if len(res) == 0:
                return np.nan
            return np.mean(np.clip(np.nan_to_num(res), 0, 1))
        else:
            raise NotImplementedError()
