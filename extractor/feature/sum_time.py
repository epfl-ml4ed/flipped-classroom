#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.feature.feature import Feature

class SumTime(Feature):

    def __init__(self, data, settings):
        super().__init__('time_in_' + settings['mode'])
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        udata = self.data.sort_values(by='date').copy()
        udata['prev_event'] = udata['event_type'].shift(1)
        udata['prev' + self.settings['mode'] + '_id'] = udata[self.settings['mode'] + '_id'].shift(1)
        udata['time_diff'] = udata['date'].diff().dt.total_seconds()
        udata = udata[(udata['prev_event'].str.contains(self.settings['mode'].title())) & (udata[self.settings['mode'] + '_id'] == udata['prev' + self.settings['mode'] + '_id'])]
        return np.sum(udata['time_diff'])
