#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.feature.feature import Feature

class TotalVideoWatched(Feature):

    def __init__(self, data, settings):
        super().__init__('total_video_watched')
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        return np.mean(self.data[self.data['event_type'].str.contains('Video.')][['video_id', 'week']].drop_duplicates().groupby(by='week').size())
