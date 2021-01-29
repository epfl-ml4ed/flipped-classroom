#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.feature.feature import Feature

class UtilizationRate(Feature):

    def __init__(self, data, settings):
        super().__init__('number_sessions')
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        taught_schedule = self.settings['course'].get_schedule()
        tmpudata = self.data[self.data['event_type'].isin(['Video.Play', 'Video.Pause', 'Video.Stop'])].sort_values(by='date')
        tmpudata['prev_event'] = tmpudata['event_type'].shift(1)
        tmpudata['prev_video_id'] = tmpudata['video_id'].shift(1)
        tmpudata['time_diff'] = tmpudata['date'].diff().dt.total_seconds()
        tmpudata = tmpudata[(tmpudata['prev_event'] == 'Video.Play') & (tmpudata['video_id'] == tmpudata['prev_video_id'])]
        return np.sum(tmpudata['time_diff']) / np.sum(taught_schedule['duration']) if np.sum(taught_schedule['duration']) > 0 else 0
