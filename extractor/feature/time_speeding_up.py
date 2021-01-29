#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_time_speeding_up

class TimeSpeedingUp(Feature):

    def __init__(self, data, settings):
        super().__init__('time_speeding_up' + ('_' + str(settings['ffunc']) if 'ffunc' in settings else ''))
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        assert self.settings['ffunc'] is not None
        return self.settings['ffunc'](get_time_speeding_up(self.data.merge(self.settings['course'].get_schedule()[['id', 'duration']], left_on=['video_id'], right_on=['id'])))