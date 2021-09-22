#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import numpy as np

import logging


'''
The (statistics) on the number of problem submissions made by a student
'''
class NumberSubmissions(Feature):

    def __init__(self, data, settings):
        super().__init__('number_submissions', data, settings)

    def compute(self):

        if not 'grade' in self.data:
            logging.debug('feature {} is invalid: no problem clickstream included'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data[self.data['event_type'].str.contains('Problem.Check') & (self.data['grade'].notnull())]
        self.data = self.data.merge(self.schedule[['id', 'grade_max']], left_on='problem_id', right_on='id')

        if 'mode' in self.settings:

            if self.settings['mode'] == 'avg':
                problem_sizes = self.data.groupby(by='problem_id').size().values
                if len(problem_sizes) == 0:
                    logging.debug('feature {} is invalid: no problem events included or grade is null for mode avg'.format(self.name))
                    return Feature.INVALID_VALUE
                return np.mean(problem_sizes)

            elif self.settings['mode'] == 'distinct':
                return len(self.data['problem_id'].unique())

            elif self.settings['mode'] == 'correct':

                if len(self.data) == 0:
                    logging.debug('feature {} is invalid: no problems included, so correct is invalid'.format(self.name))
                    return Feature.INVALID_VALUE

                correct = self.data[self.data['grade'] == self.data['grade_max']]['problem_id'].unique()
                return len(correct)

            elif self.settings['mode'] == 'perc_correct':

                if len(self.data) == 0:
                    logging.debug('feature {} is invalid: no problems included, no perc_correct is invalid'.format(self.name))
                    return Feature.INVALID_VALUE

                return len(self.data[self.data['grade'] == self.data['grade_max']]) / len(self.data['grade'])

            elif self.settings['mode'] == 'distinct_correct':

                if len(self.data) == 0:
                    logging.debug('feature {} is invalid: no problems included, so distinct_correct is invalid'.format(self.name))
                    return Feature.INVALID_VALUE

                return len(self.data[self.data['grade'] == self.data['grade_max']]['problem_id'].unique())

        return len(self.data)
