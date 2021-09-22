#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

from datetime import datetime
import pandas as pd
import numpy as np

import logging

class ContentAnticipationTime(Feature):

    def __init__(self, data, settings):
        super().__init__('content_anticipation_time', data, {**settings, **{'check_future': True}})

    def compute(self):
        assert 'course' in self.settings and 'ffunc' in self.settings and self.settings['course'].has_schedule()

        self.schedule = self.schedule[(self.schedule['week'] > self.settings['week']) & (self.schedule['type'] == 'video')]
        videos_in_future = self.schedule['id'].unique()
        if len(videos_in_future) == 0:
            logging.debug('feature {} is invalid: no videos taught in the future'.format(self.name))
            return Feature.INVALID_VALUE
        maps_schedule_date = {k:v for k, v in zip(self.schedule['id'].values, self.schedule['date'].values)}

        self.data = self.data.sort_values(by='timestamp').drop_duplicates(subset=['video_id'], keep='first')
        maps_student_date = {k:np.datetime64(datetime.utcfromtimestamp(v)) for k, v in zip(self.data['video_id'].values, self.data['timestamp'].values)}

        timings = [pd.Timedelta(maps_schedule_date[key] - maps_student_date[key]).total_seconds() // 3600 for key in maps_student_date.keys() if key in maps_schedule_date]

        if len(timings) == 0:
            logging.debug('feature {} is invalid: no match between scheduled videos and watched videos'.format(self.name))
            return Feature.INVALID_VALUE

        return self.settings['ffunc'](timings)




