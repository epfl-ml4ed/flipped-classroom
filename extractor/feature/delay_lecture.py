#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from helper.dataset.data_preparation import get_sessions, similarity_days, chi2_divergence
from extractor.feature.feature import Feature

import pandas as pd
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
        assert 'course' in self.settings and self.settings['course'].has_schedule()

        if len(self.data[self.data['event_type'].str.contains('Video')].index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.schedule = self.schedule[self.schedule['type'] == 'video']
        maps_schedule_date = {k:v for k,v in zip(self.schedule['id'].values, self.schedule['date'].values)}

        self.data = self.data.drop_duplicates(subset=['video_id'])
        maps_student_date = {k:v for k,v in zip(self.data['video_id'].values, self.data['date'].values)}

        delays = [pd.Timedelta(maps_student_date[key] - maps_schedule_date[key]).total_seconds() for key in maps_student_date.keys() if key in maps_schedule_date]

        if len(delays) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return np.mean(delays)



