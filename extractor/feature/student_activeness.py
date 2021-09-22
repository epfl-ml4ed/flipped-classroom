#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import logging


'''
The number of videos the student watched in their entirety divided by the number of loaded videos by the student
'''
class StudentActiveness(Feature):

    def __init__(self, data, settings):
        super().__init__('student_activeness', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid: empty dataframe'.format(self.name))
            return Feature.INVALID_VALUE

        videos_loaded_so_far = self.data['video_id'].unique()
        self.data['prev_event'] = self.data['event_type'].shift(1)
        self.data['prev_video'] = self.data['video_id'].shift(1)
        videos_finished_so_far = self.data[(self.data['prev_event'] == 'Video.Play') & (self.data['event_type'] == 'Video.Stop') & (self.data['prev_video'] == self.data['video_id'])]['video_id'].unique()

        if len(videos_loaded_so_far) == 0:
            logging.debug('feature {} is invalid: no video loaded by the student'.format(self.name))
            return Feature.INVALID_VALUE

        return len(set(videos_finished_so_far) & set(videos_loaded_so_far)) / len(videos_loaded_so_far)
