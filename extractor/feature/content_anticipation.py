#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import logging


'''
The number of videos covered by the student from those that are in subsequent weeks
'''
class ContentAnticipation(Feature):

    def __init__(self, data, settings):
        super().__init__('content_anticipation', data, {**settings, **{'check_future': True}})

    def compute(self):
        assert 'week' in self.settings

        self.schedule = self.schedule[(self.schedule['week'] > self.settings['week']) & (self.schedule['type'] == 'video')]
        videos_in_future = self.schedule['id'].unique()
        if len(videos_in_future) == 0:
            logging.debug('feature {} is invalid: no videos taught in the future'.format(self.name))
            return Feature.INVALID_VALUE

        videos_in_future_so_far = self.data['video_id'].unique()

        return len(set(videos_in_future_so_far) & set(videos_in_future)) / len(videos_in_future)
