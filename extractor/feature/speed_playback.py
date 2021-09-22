#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import numpy as np

import logging

'''
The (statistics) on the speed used by the student to play videos
'''
class SpeedPlayback(Feature):

    def __init__(self, data, settings):
        super().__init__('speed_playback_', data, settings)

    def compute(self):

        if len(self.data.index) == 0 or len(self.data[self.data['event_type'].str.contains('Video.')]) == 0:
            logging.debug('feature {} is invalid: empty dataframe or no video events included'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data.copy()
        self.data['new_speed'] = self.data['new_speed'].fillna(method='ffill')
        self.data['new_speed'] = self.data['new_speed'].fillna(self.data.old_speed.fillna(method='bfill'))

        self.data = self.data.dropna(subset=['new_speed'])

        new_speeds = self.data['new_speed'].values

        if np.isnan(new_speeds).all():
            logging.debug('feature {} is invalid: set to default speed'.format(self.name))
            return 1.0

        return self.settings['ffunc'](new_speeds)
