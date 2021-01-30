#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from extractor.feature.feature import Feature

from helper.dataset.data_preparation import count_events

'''
The (statistics) on the pause lenghts
'''
class PauseDuration(Feature):

    def __init__(self, data, settings):
        super().__init__('pause_duration', data, settings)

    def compute(self):
        assert 'ffunc' in self.settings

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.data['prev_event'] = self.data['event_type'].shift(1)
        self.data['prev_video_id'] = self.data['video_id'].shift(1)
        self.data['time_diff'] = self.data['date'].diff().dt.total_seconds()
        self.data = self.data.dropna(subset=['time_diff'])

        pause_durations = self.data[(self.data['prev_event'] == 'Video.Pause') & (self.data['video_id'] == self.data['prev_video_id'])]['time_diff'].values
        return self.settings['ffunc'](pause_durations)
