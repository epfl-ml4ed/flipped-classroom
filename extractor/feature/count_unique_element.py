#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import logging

'''
The number of unique elements (either videos or problems)
'''
class CountUniqueElement(Feature):

    def __init__(self, data, settings):
        super().__init__('count_unique_elements', data, settings)

    def compute(self):
        assert 'course' in self.settings and self.settings['course'].has_schedule() and 'type' in self.settings

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return len(self.data.drop_duplicates(subset=[self.settings['type'] + '_id']))
