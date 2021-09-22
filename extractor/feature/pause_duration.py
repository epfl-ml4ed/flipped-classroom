#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature
from extractor.feature.time import Time

import logging


'''
The (statistics) on the pause lenghts
'''
class PauseDuration(Feature):

    def __init__(self, data, settings):
        super().__init__('pause_duration', data, settings)

    def compute(self):
        assert 'ffunc' in self.settings

        return Time(self.data, {**self.settings, **{'type': 'Video.Pause'}}).compute()