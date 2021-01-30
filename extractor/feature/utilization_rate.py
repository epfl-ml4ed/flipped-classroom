#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging

from extractor.feature.feature import Feature

'''
The utilization rate of a student s on a given week c since the beginning of the course is the proportion
of video play time activity of the student over the sum of video lengths for all videos up to week c.
'''
class UtilizationRate(Feature):

    def __init__(self, data, settings):
        super().__init__('utilization_rate', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.data['prev_event'] = self.data['event_type'].shift(1)
        self.data['prev_video_id'] = self.data['video_id'].shift(1)
        self.data['time_diff'] = self.data['date'].diff().dt.total_seconds()
        self.data = self.data.dropna(subset=['time_diff'])
        self.data = self.data[(self.data['time_diff'] >= Feature.TIME_MIN) & (self.data['time_diff'] <= Feature.TIME_MAX)]

        time_intervals = self.data[(self.data['prev_event'] == 'Video.Play') & (self.data['video_id'] == self.data['prev_video_id'])]['time_diff'].values
        sum_time_intervals = np.sum(time_intervals[(time_intervals >= Feature.TIME_MIN) & (time_intervals <= Feature.TIME_MAX)])

        self.schedule = self.schedule[self.schedule['type'] == 'video']
        sum_video_lengths = np.sum(self.schedule['duration'].values)

        if sum_video_lengths == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return sum_time_intervals / sum_video_lengths
