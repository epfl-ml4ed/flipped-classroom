#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging

class Feature():

    INVALID_VALUE = np.nan
    TIME_MIN = 3.0
    TIME_MAX = 3600

    WEEKEND = [5, 6]
    WEEKDAY = [0, 1, 2, 3, 4]

    def __init__(self, name, data, settings):
        assert 'timeframe' in settings
        self.settings = settings
        self.name = self.rename(name)
        self.data = self.filter(data).copy().sort_values(by='date')
        self.schedule = self.filter(settings['course'].get_schedule())

    def rename(self, name):
        assert self.name is not None and self.name is not ''
        if 'mode' in self.settings:
            name += '_' + self.settings['mode']
        if 'type' in self.settings:
            name += '_' + self.settings['type']
        if 'ffunc' in self.settings:
            name += '_' + self.settings['ffunc']
        return name

    def filter(self, data):
        if self.settings['timeframe'] == 'full':
            return data

        if self.settings['timeframe'] == 'eq-week':
            assert 'week' in self.settings
            logging.info('framing data on a {} timeframe'.format(self.settings['timeframe']))
            return data[data['week'] == self.settings['week']]

        if self.settings['timeframe'] == 'lq-week':
            assert 'week' in self.settings
            logging.info('framing data on a {} timeframe'.format(self.settings['timeframe']))
            return data[data['week'] <= self.settings['week']]

        logging.info('no framing is applied')
        return data

    def compute(self):
        return None

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data