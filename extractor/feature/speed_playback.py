#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from extractor.feature.feature import Feature

'''
The (statistics) on the speed used by the student to play videos
'''
class SpeedPlayback(Feature):

    def __init__(self, data, settings):
        super().__init__('time_in_', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data[self.data['event_type'].str.contains('Video.SpeedChange')]
        self.data['new_speed'] = self.data['new_speed'].fillna(method='ffill')
        self.data = self.data.dropna(subset=['new_speed'])
        return self.settings['ffunc'](self.data['new_speed'].values)
