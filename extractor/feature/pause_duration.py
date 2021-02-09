#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from extractor.feature.feature import Feature
from extractor.feature.time import Time

from helper.dataset.data_preparation import count_events

'''
The (statistics) on the pause lenghts
'''
class PauseDuration(Feature):

    def __init__(self, data, settings):
        super().__init__('pause_duration', data, settings)

    def compute(self):
        assert 'ffunc' in self.settings

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return Time(self.data, {**self.settings, **{'type': 'Video.Pause'}}).compute()