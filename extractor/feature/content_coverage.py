#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import logging


'''
The number of videos covered by the student since the beginning of the course
'''
class ContentCoverage(Feature):

    def __init__(self, data, settings):
        super().__init__('content_coverage', data, settings)

    def compute(self):
        assert 'week' in self.settings

        self.schedule = self.schedule[self.schedule['type'] == 'video']
        videos_to_cover = self.schedule['id'].unique()
        if len(videos_to_cover) == 0:
            logging.debug('feature {} is invalid: no videos taught'.format(self.name))
            return Feature.INVALID_VALUE

        videos_so_far = self.data['video_id'].unique()

        return len(set(videos_so_far) & set(videos_to_cover)) / len(videos_to_cover)
