#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.feature.feature import Feature
from helper.dataset.data_preparation import get_sessions

class CountNGrams(Feature):

    def __init__(self, data, settings):
        super().__init__('count_ngrams')
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return np.zeros(len(self.settings['perms'])).tolist()
        sessions = get_sessions(self.data.sort_values(by='date'), self.settings['max_session_length'], self.settings['min_actions'])
        maps = {c:i for i, c in enumerate(self.settings['perms'])}
        counts = np.zeros(len(self.settings['perms']))

        for events in sessions['event']:
            s = events.split(',')
            if len(s) >= self.settings['ngram']:
                for i in range(len(s) - self.settings['ngram'] + 1):
                    c = (s[i],)
                    for j in range(i+1, i + self.settings['ngram']):
                        c += (s[j],)
                    counts[maps[c]] += 1

        return list(counts)
