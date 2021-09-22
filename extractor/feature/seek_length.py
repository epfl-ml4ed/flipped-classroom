#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import logging


'''
The (statistics) length of the seek events
'''
class SeekLength(Feature):

    def __init__(self, data, settings):
        super().__init__('seek_length', data, settings)

    def compute(self):
        assert 'ffunc' in self.settings

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid: empty dataframe'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data[self.data['event_type'] == 'Video.Seek']
        seek_lengths = abs(self.data['old_time'] - self.data['new_time']).values

        if len(seek_lengths) == 0:
            logging.debug('feature {} is invalid: no video seek events included'.format(self.name))
            return Feature.INVALID_VALUE

        return self.settings['ffunc'](seek_lengths)