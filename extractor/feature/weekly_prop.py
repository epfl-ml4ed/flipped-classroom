#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

from helper.dataset.data_preparation import get_weekly_prop_watched, get_weekly_prop_replayed, get_weekly_prop_interrupted

class WeeklyProp(Feature):

    def __init__(self, data, settings):
        super().__init__('weekly_prop' + ('_' + settings['type'] if 'type' in settings else '') + ('_' + str(settings['ffunc']) if 'ffunc' in settings else ''))
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        if self.settings['type'] == 'replayed':
            return self.settings['ffunc'](get_weekly_prop_replayed(self.data, self.settings))
        if self.settings['type'] == 'watched':
            return self.settings['ffunc'](get_weekly_prop_watched(self.data, self.settings))
        if self.settings['type'] == 'interrupted':
            return self.settings['ffunc'](get_weekly_prop_interrupted(self.data, self.settings))
        raise NotImplementedError()
