#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import numpy as np
import logging

'''
The amount of time passed reflecting on pause during a video
'''
class StudentThoughtfulness(Feature):

    def __init__(self, data, settings):
        super().__init__('student_thoughtfulness', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data[self.data['event_type'].isin(['Video.Play', 'Video.Pause'])]
        self.data['prev_event'] = self.data['event_type'].shift(1)
        self.data['prev_video'] = self.data['video_id'].shift(1)
        self.data = self.data[(self.data['prev_event'] == 'Video.Pause') & (self.data['event_type'] == 'Video.Play') & (self.data['prev_video'] == self.data['video_id'])]
        self.data['time_diff'] = self.data['date'].diff().dt.total_seconds()
        self.data = self.data.dropna(subset=['time_diff'])
        self.data = self.data[(self.data['time_diff'] >= Feature.TIME_MIN) & (self.data['time_diff'] <= Feature.TIME_MAX)]

        if np.sum(self.data['time_diff'].values) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return 1 - 1 / np.sum(self.data['time_diff'].values)
