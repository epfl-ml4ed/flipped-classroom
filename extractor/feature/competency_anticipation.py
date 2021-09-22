#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import logging


'''
The number of problems covered by the student from those that are in subsequent weeks
'''
class CompetencyAnticipation(Feature):

    def __init__(self, data, settings):
        super().__init__('competency_anticipation', data, {**settings, **{'check_future': True}})

    def compute(self):
        assert 'week' in self.settings

        self.schedule = self.schedule[(self.schedule['week'] > self.settings['week']) & (self.schedule['type'] == 'problem')]
        problems_in_future = self.schedule['id'].unique()
        if len(problems_in_future) == 0:
            logging.debug('feature {} is invalid: no problems taught in the future'.format(self.name))
            return Feature.INVALID_VALUE

        problems_so_far = self.data['problem_id'].unique()

        return len(set(problems_so_far) & set(problems_in_future)) / len(problems_in_future)
