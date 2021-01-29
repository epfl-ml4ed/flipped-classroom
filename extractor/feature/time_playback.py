#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_sessions

class TimePlayback(Feature):

    def __init__(self, data, settings):
        super().__init__('time_playback' + ('_' + settings['mode'] if 'mode' in settings else '') + ('_' + str(settings['ffunc']) if 'ffunc' in settings else ''))
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        if 'Backward' in self.settings['mode']:
            udata = self.data[(self.data['event_type'] == 'Video.Seek') & (self.data['old_time'] > self.data['new_time'])]
        elif 'Forward' in self.settings['mode']:
            udata = self.data[(self.data['event_type'] == 'Video.Seek') & (self.data['old_time'] < self.data['new_time'])]
        else:
            raise NotImplementedError()
        return self.settings['ffunc'](udata['old_time'] - udata['new_time'])
