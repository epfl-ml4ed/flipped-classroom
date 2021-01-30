#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging
import math

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
            if video_id not in maps_duration:
                continue

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
                maps_spent[video_id] = 1 if np.count_nonzero(video_completed) >= len(video_completed) * 0.90 else 0
            else:
                raise NotImplementedError()

        return maps_spent

    def compute(self):
        assert 'course' in self.settings and self.settings['course'].has_schedule()

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        self.data = self.data.sort_values(by=['video_id', 'current_time'])
        self.data['current_time'] = self.data.apply(lambda row: row['old_time'] if not math.isnan(row['old_time']) else row['current_time'], axis=1)
        self.data['prev_event'] = self.data['event_type'].shift(1)
        self.data['prev_video_id'] = self.data['video_id'].shift(1)
        self.data['prev_current_time'] = self.data['current_time'].shift(1)
        self.data['time_diff'] = self.data['date'].diff().dt.total_seconds()
        self.data = self.data.dropna(subset=['time_diff'])
        self.data = self.data.dropna(subset=['current_time'])
        self.data = self.data[(self.data['time_diff'] >= Feature.TIME_MIN) & (self.data['time_diff'] <= Feature.TIME_MAX)]

        maps_duration = {v:d for v, d in zip(self.schedule['id'].values, self.schedule['duration'].values)}

        if 'mode' in self.settings:

            if self.settings['mode'] in ['completed', 'played', 'entirety']:
                self.data = self.data[(self.data['prev_event'].str.contains(self.settings['type'])) & (self.data['video_id'] == self.data['prev_video_id'])]
                completion = self.find_completion(maps_duration)
                if len(completion) == 0:
                    logging.debug('feature {} is invalid'.format(self.name))
                    return Feature.INVALID_VALUE
                return np.mean(np.fromiter(completion.values(), dtype=float))

            elif self.settings['mode'] == 'time':
                self.data = self.data[(self.data['prev_event'].str.contains(self.settings['type'])) & (self.data['video_id'] == self.data['prev_video_id'])]
                if 'backward' in self.settings['phase']:
                    self.data = self.data[self.data['current_time'] < self.data['prev_current_time']]
                else:
                    self.data = self.data[self.data['current_time'] > self.data['prev_current_time']]
                self.data['diff_current_time'] = np.abs(self.data['current_time'] - self.data['prev_current_time'])
                maps_spent = {v:d for v, d in zip(self.data.groupby(by='video_id').sum().index, self.data.groupby(by='video_id').sum()['diff_current_time'])}
                if len(maps_spent) == 0:
                    logging.debug('feature {} is invalid'.format(self.name))
                    return Feature.INVALID_VALUE
                return np.mean(np.fromiter(maps_spent.values(), dtype=float))

        self.data = self.data[(self.data['prev_event'].str.contains(self.settings['type'])) & (self.data['video_id'] == self.data['prev_video_id'])]
        maps_spent = {v:d for v, d in zip(self.data.groupby(by='video_id').sum().index, self.data.groupby(by='video_id').sum()['time_diff'].values)}

        percentages = np.zeros(len(maps_duration))
        for i, (video_id, video_duration) in enumerate(maps_duration.items()):
            if video_id in maps_spent:
                percentages[i] = maps_spent[video_id] / video_duration
        
        if len(percentages) == 0:
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE
        
        return np.mean(percentages)

