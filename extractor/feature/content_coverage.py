#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import numpy as np
import logging

'''
The number of videos covered by the student since the beginning of the course
'''
class ContentCoverage(Feature):

    def __init__(self, data, settings):
        super().__init__('content_coverage', data, {**settings, **{'timeframe': 'full'}})

    def compute(self):
        assert 'week' in self.settings

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.schedule = self.schedule[(self.schedule['week'] <= self.settings['week']) & (self.schedule['type'] == 'video')]
        self.data = self.data[self.data['week'] <= self.settings['week']]

        videos_to_cover = self.schedule['id'].unique()
        videos_so_far = self.data['video_id'].unique()

        if len(videos_to_cover) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return len(set(videos_so_far) & set(videos_to_cover)) / len(videos_to_cover)
