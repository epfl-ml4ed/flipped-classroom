#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from extractor.feature.total_clicks import TotalClicks
from extractor.feature.feature import Feature

'''
The ration between the number of clicks in the weekend and the weekdays
'''
class RatioClicksWeekendDay(Feature):

    def __init__(self, data, settings):
        super().__init__('ratio_clicks_weekend_day', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        clicks_weekday = TotalClicks(self.data, {**self.settings, **{'mode':'weekday'}}).compute()
        clicks_weekend = TotalClicks(self.data, {**self.settings, **{'mode':'weekend'}}).compute()

        if clicks_weekend == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return clicks_weekday / clicks_weekend
