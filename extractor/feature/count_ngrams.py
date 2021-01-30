#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_sessions

'''
The counts of ngrams in the sessions performed by a student
'''
class CountNGrams(Feature):

    def __init__(self, data, settings):
        super().__init__('count_ngrams', data, settings)

    def compute(self):
        assert 'ngram' in self.settings and 'perms' in self.settings

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return (np.zeros(len(self.settings['perms'])) + Feature.INVALID_VALUE).tolist()

        sessions = get_sessions(self.data.sort_values(by='date'), self.schedule['duration'].max())
        maps = {c:i for i,c in enumerate(self.settings['perms'])}

        counts = np.zeros(len(self.settings['perms']))
        for events in sessions['event']:
            s = events.split(',')
            if len(s) >= self.settings['ngram']:
                for i in range(len(s) - self.settings['ngram']):
                    c = (s[i],)
                    for j in range(i+1, i + self.settings['ngram']):
                        c += (s[j],)
                    counts[maps[c]] += 1

        return list(counts)
