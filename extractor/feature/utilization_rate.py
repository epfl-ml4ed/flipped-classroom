#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging

from extractor.feature.feature import Feature
from extractor.feature.time import Time

'''
The utilization rate of a student s on a given week c since the beginning of the course is the proportion
of video play time activity of the student over the sum of video lengths for all videos up to week c.
Note: since we don't have access directly to the time spent by a user watching videos we have to infer it.
The current model measures the time between every Video.Play action and the following action.
'''
class UtilizationRate(Feature):

    def __init__(self, data, settings):
        super().__init__('utilization_rate', data, settings)

    def compute(self):
        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        sum_time_intervals = Time(self.data, {**self.settings, **{'type': 'Video.Play', 'ffunc': np.sum}}).compute()

        self.schedule = self.schedule[self.schedule['type'] == 'video']
        sum_video_lengths = np.sum(self.schedule['duration'].values)
        if sum_video_lengths == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return sum_time_intervals / sum_video_lengths
