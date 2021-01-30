#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import numpy as np
import logging

'''
The number of problems for that week that have been passed by the student
'''
class CompetencyAlignment(Feature):

    def __init__(self, data, settings):
        super().__init__('competency_alignment', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data[self.data['event_type'].str.contains('Problem.Check') & (self.data['grade'].notnull())]
        self.data = self.data.merge(self.schedule, left_on='problem_id', right_on='id')
        learnt_problems = self.data[self.data['grade'] == self.data['grade_max']]['problem_id'].unique()
        taught_problems = self.schedule[self.schedule['type'] == 'problem']['id'].unique()

        if len(taught_problems) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return len(set(learnt_problems) & set(learnt_problems)) / len(taught_problems)
