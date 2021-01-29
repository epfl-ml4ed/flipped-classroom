#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

class SpeedPlayback(Feature):

    def __init__(self, data, settings):
        super().__init__('time_in_' + str(settings['ffunc']))
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        return self.settings['ffunc'](self.data[self.data['event_type'].str.contains('Video.SpeedChange')]['new_speed'])
