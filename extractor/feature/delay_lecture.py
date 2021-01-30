#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from helper.dataset.data_preparation import get_sessions, similarity_days, chi2_divergence
from extractor.feature.feature import Feature

from scipy.spatial.distance import jensenshannon
import numpy as np
import logging

'''
Some students watch the lecture right after it is released whereas others postpone watching lectures or submitting assignments.
Therefore some users are regular not because of a weekly routine, but they follow the schedule of the course. To capture adherence
to the course schedule, we define DLV measure as the average delay in viewing video lectures.
'''
class DelayLecture(Feature):

    def __init__(self, data, settings):
        super().__init__('delay_lecture', data, settings)

    def compute(self):
        assert 'mode' in self.settings and self.settings['timeframe'] is not 'eq-week' and self.settings['week'] > 0
        assert 'course' in self.settings and self.settings['course'].has_schedule()

        if len(self.data[self.data['event_type'].str.contains('Video')].index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.schedule = self.schedule[self.schedule['type'] == 'video']
        maps_schedule_date = {k:v for k,v in zip(self.schedule['id'].index, self.schedule['date'].values)}

        self.data = self.data.drop_duplicates(subset=['id'])
        maps_student_date = {k:v for k,v in zip(self.schedule['id'].index, self.data['date'].values)}
        return np.mean([(maps_student_date[key] - maps_schedule_date[key]).dt.total_seconds() for key in maps_student_date.keys()])



