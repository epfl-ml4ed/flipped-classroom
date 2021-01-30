#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import numpy as np
import logging

'''
The number of problems covered by the student since the beginning of the course
'''
class CompetencyCoverage(Feature):

    def __init__(self, data, settings):
        super().__init__('competency_coverage', data, {**settings, **{'timeframe': 'full'}})

    def compute(self):
        assert 'week' in self.settings

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.schedule = self.schedule[(self.schedule['week'] <= self.settings['week']) & (self.schedule['type'] == 'problem')]
        self.data = self.data[self.data['week'] <= self.settings['week']]

        problems_to_cover = self.schedule['id'].unique()
        problems_so_far = self.data['problem_id'].unique()

        if len(problems_to_cover) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return len(set(problems_so_far) & set(problems_to_cover)) / len(problems_to_cover)
