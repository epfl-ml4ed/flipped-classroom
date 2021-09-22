#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np

from extractor.feature.feature import Feature

'''
The total number of clicks provided by a student (on a given set of days) (on a given type of events)
'''
class TextLength(Feature):

    def __init__(self, data, settings):
        super().__init__('text_length', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid: empty dataframe'.format(self.name))
            return Feature.INVALID_VALUE

        data = self.data

        res = 0

        if 'type' in self.settings:
            data = data[self.data['event_type'].str.contains(self.settings['type'].title())]

        if 'field' in self.settings:
            df_text = data[self.settings['field']][data[self.settings['field']].notna()]
            if len(df_text) > 0:
                length = df_text.str.len()

                if self.settings['ffunc'] == 'avg':
                    res = np.mean(length)

                elif self.settings['ffunc'] == 'max':
                    res = np.max(length)

        return res