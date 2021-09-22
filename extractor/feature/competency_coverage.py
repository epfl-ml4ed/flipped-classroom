#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import logging


'''
The number of problems covered by the student since the beginning of the course
'''
class CompetencyCoverage(Feature):

    def __init__(self, data, settings):
        super().__init__('competency_coverage', data, settings)

    def compute(self):
        assert 'week' in self.settings

        self.schedule = self.schedule[self.schedule['type'] == 'problem']
        problems_to_cover = self.schedule['id'].unique()
        if len(problems_to_cover) == 0:
            logging.debug('feature {} is invalid: no problems taught in that period'.format(self.name))
            return Feature.INVALID_VALUE

        problems_so_far = self.data['problem_id'].unique()

        return len(set(problems_so_far) & set(problems_to_cover)) / len(problems_to_cover)
