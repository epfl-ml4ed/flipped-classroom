#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_sessions

class TimePlaybackChange(Feature):

    def __init__(self, data, settings):
        super().__init__('time_playback_change' + ('_' + str(settings['ffunc']) if 'ffunc' in settings else ''))
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        udata = self.data[self.data['event_type'] == 'Video.SpeedChange'].copy()
        udata['prev_event'] = udata['event_type'].shift(1)
        udata['prev_video_id'] = udata['video_id'].shift(1)
        udata['time_diff'] = udata['date'].diff().dt.total_seconds()
        udata = udata[udata['video_id'] == udata['prev_video_id']]
        return self.settings['ffunc'](udata['time_diff'])

