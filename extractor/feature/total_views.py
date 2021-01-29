#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

class TotalViews(Feature):

    def __init__(self, data, settings):
        super().__init__('total_views' + ('_' + settings['type'] if 'type' in settings else ''))
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        return len(self.data.drop_duplicates(subset=[self.settings['type'] + '_id', 'week', 'weekday']))
