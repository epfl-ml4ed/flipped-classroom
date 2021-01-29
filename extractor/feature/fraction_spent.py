#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from extractor.feature.feature import Feature

class FractionSpent(Feature):

    def __init__(self, data, settings):
        super().__init__('fraction_spent' + ('_' + settings['type'] if 'type' in settings else '') + ('_' + settings['mode'].lower().replace('.', '_') if 'mode' in settings else ''))
        self.data = data if 'week' not in settings else (data[data['week'] == settings['week']] if settings['timeframe'] == 'week' else data[data['week'] <= settings['week']])
        self.settings = settings

    def compute(self):
        if len(self.data.index) == 0:
            return 0.0
        course_schedule = self.settings['course'].get_schedule()
        course_schedule = course_schedule[course_schedule['type'] == 'video']
        udata = self.data.sort_values(by=['video_id', 'current_time'])
        udata['prev_event'] = udata['event_type'].shift(1)
        udata['prev_video_id'] = udata['video_id'].shift(1)
        udata['time_diff'] = udata['date'].diff().dt.total_seconds()

        if 'type' in self.settings:
            if self.settings['type'] == 'repeated_perc_time':
                udata = udata[(udata['prev_event'].str.contains(self.settings['mode'])) & (udata['video_id'] == udata['prev_video_id'])]
                maps = {v: d for v, d in zip(udata.groupby(by='video_id').sum().index, udata.groupby(by='video_id').sum()['current_time'])}
            elif self.settings['type'].startswith('perc_time'):
                udata = udata[(udata['prev_event'].str.contains(self.settings['mode'])) & (udata['video_id'] == udata['prev_video_id'])]
                udata = udata.drop_duplicates(subset=['video_id'], keep='last')
                maps = {v: d for v, d in zip(udata.groupby(by='video_id').sum().index, udata.groupby(by='video_id').sum()['current_time'])}
        else:
            udata = udata[(udata['prev_event'].str.contains('Video.')) & (udata['video_id'] == udata['prev_video_id'])]
            maps = {v: d for v, d in zip(udata.groupby(by='video_id').sum().index, udata.groupby(by='video_id').sum()['time_diff'])}

        if not ('type' in self.settings and self.settings['type'] == 'perc_time_entire_video'):
            ratio = []
            for videoID, timeSchedule in zip(course_schedule['id'], course_schedule['duration']):
                ratio.append((maps[videoID] if videoID in maps else 0) / timeSchedule)
            return np.mean(ratio)

        counter = 0
        for videoID, timeSchedule in zip(course_schedule['id'], course_schedule['duration']):
            counter += (1 if videoID in maps and abs(maps[videoID] - timeSchedule) < 15.0 else 0)
        return counter

