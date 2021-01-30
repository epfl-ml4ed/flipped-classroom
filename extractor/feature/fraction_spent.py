#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging

from extractor.feature.feature import Feature

'''
The (statistics) on the amount of playing/pausing with a video with respect to its length
'''
class FractionSpent(Feature):

    def __init__(self, data, settings):
        super().__init__('fraction_spent', data, settings)

    def find_completion(self, maps_duration):
        maps_spent = {}
        for video_id, video_data in self.data.groupby(by='video_id'):
            video_completed = np.zeros(int(maps_duration[video_id]))

            for index, row in video_data.iterrows():
                start_current_time = int(row['prev_current_time']) if row['prev_current_time'] < row['current_time'] else int(row['current_time'])
                end_current_time = int(row['prev_current_time']) if row['prev_current_time'] > row['current_time'] else int(row['current_time'])
                video_completed[start_current_time:end_current_time] += 1

            if self.settings['mode'].startswith('completed'):
                maps_spent[video_id] = np.count_nonzero(video_completed) / len(video_completed)
            elif self.settings['mode'].startswith('played'):
                maps_spent[video_id] = np.sum(video_completed)
            elif self.settings['mode'].startswith('entirety'):
                maps_spent[video_id] = 1 if np.count_nonzero(video_completed) == len(video_completed) else 0
            else:
                raise NotImplementedError
        return maps_spent

    def compute(self):
        assert 'course' in self.settings and self.settings['course'].has_schedule()

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data.sort_values(by=['video_id', 'current_time'])
        self.data['current_time'] = self.data.apply(lambda row: row['old_time'] if row['old_time'] is not None else row['current_time'], axis=1)
        self.data['prev_event'] = self.data['event_type'].shift(1)
        self.data['prev_video_id'] = self.data['video_id'].shift(1)
        self.data['prev_current_time'] = self.data['current_time'].shift(1)
        self.data['time_diff'] = self.data['date'].diff().dt.total_seconds()
        self.data = self.data.dropna(subset=['time_diff'])
        self.data = self.data[(self.data['time_diff'] >= Feature.TIME_MIN) & (self.data['time_diff'] <= self.schedule['duration'].max())]

        maps_duration = {v:d for v, d in zip(self.schedule['id'].values, self.schedule['duration'].values)}

        if 'mode' in self.settings:

            if self.settings['mode'] in ['completed', 'played', 'entirety']:
                self.data = self.data[(self.data['prev_event'].str.contains(self.settings['type'])) & (self.data['video_id'] == self.data['prev_video_id'])]
                return np.mean(self.find_completion(maps_duration).values())
            elif self.settings['mode'] == 'time':
                self.data = self.data[(self.data['prev_event'].str.contains(self.settings['type'])) & (self.data['video_id'] == self.data['prev_video_id'])]
                if 'backward' in self.settings['phase']:
                    self.data = self.data[self.data['current_time'] < self.data['prev_current_time']]
                else:
                    self.data = self.data[self.data['current_time'] > self.data['prev_current_time']]
                self.data['diff_current_time'] = np.abs(self.data['current_time'] - self.data['prev_current_time'])
                maps_spent = {v:d for v, d in self.data.groupby(by='video_id').sum().index, self.data.groupby(by='video_id').sum()['diff_current_time']}
                return np.mean(maps_spent.values())

        self.data = self.data[(self.data['prev_event'].str.contains(self.settings['type'])) & (self.data['video_id'] == self.data['prev_video_id'])]
        maps_spent = {v:d for v, d in zip(self.data.groupby(by='video_id').sum().index, self.data.groupby(by='video_id').sum()['time_diff'].values)}

        percentages = np.zeros(len(maps_spent))
        for i, (video_id, video_spent) in enumerate(maps_spent.items()):
            percentages[i] = video_spent / maps_duration[video_id]

        return np.mean(percentages)

