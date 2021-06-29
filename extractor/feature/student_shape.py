#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

from itertools import groupby
import numpy as np
import logging

'''
The amount of problems passed by the student at the first tentative in a row
'''
class StudentShape(Feature):

    def __init__(self, data, settings):
        super().__init__('student_shape', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        if not 'grade' in self.data:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data[self.data['event_type'].str.contains('Problem.Check') & (self.data['grade'].notnull())].sort_values(by=['problem_id', 'date'])
        self.data = self.data.merge(self.schedule, left_on='problem_id', right_on='id')

        if len(self.data) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        first_tentative = self.data.drop_duplicates(subset=['problem_id'], keep='first')
        grades_first_tentative = np.where(first_tentative['grade'].values == first_tentative['grade_max'].values, 1, 0)
        count_dups = np.array([sum(1 for _ in group) for _, group in groupby(grades_first_tentative)])
        unique_dup = np.array([x[0] for x in groupby(grades_first_tentative)]) / 100.
        probs = count_dups / np.sum(count_dups)
        return np.sum(probs[np.where(unique_dup > 0)[0]])
