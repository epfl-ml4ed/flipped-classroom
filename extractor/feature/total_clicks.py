#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

class TotalClicks(Feature):

    def __init__(self, data, settings):
        super().__init__('total_clicks' + ('_' + settings['mode'] if 'mode' in settings else '') + ('_' + settings['type'] if 'type' in settings else ''))
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        data = self.data
        if 'mode' in self.settings:
            if self.settings['mode'] == 'weekend':
                data = self.data[self.data['weekday'].isin([5, 6])]
            if self.settings['mode'] == 'weekday':
                data = self.data[self.data['weekday'] < 5]
            if 'type' in self.settings:
                data = self.data[self.data['event_type'].str.contains(self.settings['type'].title())]
        return len(data.index)
