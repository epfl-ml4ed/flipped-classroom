#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import numpy as np
import logging

'''
The number of submissions for problems related to the corresponding grade
'''
class CompetencyStrength(Feature):

    def __init__(self, data, settings):
        super().__init__('competency_strength', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data[self.data['event_type'].str.contains('Problem.Check') & (self.data['grade'].notnull())]
        self.data = self.data.merge(self.schedule, left_on='problem_id', right_on='id')

        self.data = self.data.groupby(by='problem_id').max()

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        no_submissions = self.data['submission_number'].values
        no_grades = self.data['grade'].values / self.data['grade_max'].values
        return np.mean((1 / no_submissions) * no_grades)
