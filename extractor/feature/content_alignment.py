#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import logging


'''
The number of videos for that week that have been watched by the student
'''
class ContentAlignment(Feature):

    def __init__(self, data, settings):
        super().__init__('content_alignment', data, settings)

    def compute(self):

        taught_videos = self.schedule[self.schedule['type'] == 'video']['id'].unique()
        if len(taught_videos) == 0:
            logging.debug('feature {} is invalid: no video taught in that period'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data[self.data['event_type'].str.contains('Video')]
        learnt_videos = self.data['video_id'].unique()

        return len(set(learnt_videos) & set(taught_videos)) / len(taught_videos)
