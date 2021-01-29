#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_sessions

class AttendanceRate(Feature):

    def __init__(self, data, settings):
        super().__init__('number_sessions')
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        taught_schedule = self.settings['course'].get_schedule()
        learnt_schedule = self.data.drop_duplicates(subset=['video_id'], keep='first')
        return len(set(learnt_schedule['video_id']) & set(taught_schedule['id'])) / len(taught_schedule) if len(taught_schedule) > 0 else 0
