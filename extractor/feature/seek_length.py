#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_seek_length

class SeekLength(Feature):

    def __init__(self, data, settings):
        super().__init__('seek_length' + ('_' + str(settings['ffunc']) if 'ffunc' in settings else ''))
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0

        assert self.settings['ffunc'] is not None

        slice = get_seek_length(self.data)
        if len(slice) > 0:
            return self.settings['ffunc'](slice)

        return 0.0