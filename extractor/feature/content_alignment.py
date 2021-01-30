#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import numpy as np
import logging

'''
The number of videos for that week that have been watched by the student
'''
class ContentAlignment(Feature):

    def __init__(self, data, settings):
        super().__init__('content_alignment', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data[self.data['event_type'].str.contains('Video') & (self.data['grade'].notnull())]
        learnt_videos = self.data['video_id'].unique()
        taught_videos = self.schedule[self.schedule['type'] == 'video']['id'].unique()

        if len(taught_videos) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return len(set(learnt_videos) & set(taught_videos)) / len(taught_videos)
