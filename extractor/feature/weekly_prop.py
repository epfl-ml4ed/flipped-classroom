#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from extractor.feature.feature import Feature

from helper.dataset.data_preparation import get_weekly_prop_watched, get_weekly_prop_replayed, get_weekly_prop_interrupted

'''
The proportion of videos that are replayed, watched, or interrupted
'''
class WeeklyProp(Feature):

    def __init__(self, data, settings):
        super().__init__('weekly_prop', data, settings)

    def compute(self):
        assert 'type' in self.settings and 'ffunc' in self.settings and 'course' in self.settings and self.settings['course'].has_schedule()

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        if self.settings['type'] == 'replayed':
            weekly_prop_data = self.filter(get_weekly_prop_replayed(self.data, self.settings))[0].values
            if len(weekly_prop_data) == 0:
                logging.debug('feature {} is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            return self.settings['ffunc'](weekly_prop_data)

        if self.settings['type'] == 'watched':
            weekly_prop_data = self.filter(get_weekly_prop_watched(self.data, self.settings))[0].values
            if len(weekly_prop_data) == 0:
                logging.debug('feature {} is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            return self.settings['ffunc'](weekly_prop_data)

        if self.settings['type'] == 'interrupted':
            weekly_prop_data = self.filter(get_weekly_prop_interrupted(self.data, self.settings))[0].values
            if len(weekly_prop_data) == 0:
                logging.debug('feature {} is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            return self.settings['ffunc'](weekly_prop_data)

        raise NotImplementedError()
