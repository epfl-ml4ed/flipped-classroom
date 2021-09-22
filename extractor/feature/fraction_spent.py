#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature
from extractor.feature.time import Time

import numpy as np
import math

import logging


'''
The (statistics) on the amount of playing/pausing with a video with respect to its length

Fraction played (type = Video.Play): The amount of the video that the user played, with repetition, relative to its length.
Fraction paused (type = Video.Pause): The fraction of time the user spent paused on the video, relative to its length.
Fraction spent watching (mode = played, type = Video.Play): The fraction of (real) time the user spent playing the video,
relative to its length.
Fraction completed (mode = completed, type = Video.Play): The percentage of the video that the user played, not counting
repeated play position intervals; hence, it must be between 0 and 1.
'''


class FractionSpent(Feature):

    def __init__(self, data, settings):
        super().__init__('fraction_spent', data, settings)

    def find_completion(self, video_data, maps_duration):
        maps_spent = {}

        for video_id, video_data in video_data.groupby(by='video_id'):

            if video_id not in maps_duration or math.isnan(maps_duration[video_id]):
                continue

            video_completed = np.zeros(int(maps_duration[video_id]))

            for index, row in video_data.iterrows():
                video_completed[int(row['prev_current_time']):int(row['current_time'])] += 1

            if self.settings['mode'].startswith('completed'):
                maps_spent[video_id] = np.count_nonzero(video_completed) / len(video_completed)

            elif self.settings['mode'].startswith('entirety'):
                maps_spent[video_id] = 1 if np.count_nonzero(video_completed) >= len(video_completed) * 0.90 else 0

            else:
                raise NotImplementedError()

        return maps_spent

    def compute(self):
        assert 'course' in self.settings and self.settings['course'].has_schedule() and 'type' in self.settings

        if len(self.data.index) == 0:
            logging.debug('feature {} is invalid: empty dataframe'.format(self.name))
            return Feature.INVALID_VALUE

        maps_duration = {v: d for v, d in zip(self.schedule['id'].values, self.schedule['duration'].values)}

        if self.settings['mode'] == 'ratio_duration':

            percentages = []
            for i, (video_id, video_data) in enumerate(self.data.groupby(by='video_id')):
                if video_id not in maps_duration or math.isnan(maps_duration[video_id]):
                    continue
                video_data['prev_event'] = video_data['event_type'].shift(1)
                video_data['prev_video_id'] = video_data['video_id'].shift(1)
                video_data['time_diff'] = video_data['date'].diff().dt.total_seconds()
                video_data = video_data.dropna(subset=['time_diff'])
                video_data = video_data[(video_data['time_diff'] >= Feature.TIME_MIN) & (video_data['time_diff'] <= Feature.TIME_MAX)]
                time_intervals = video_data[video_data['prev_event'] == 'Video.Play']['time_diff'].values
                time_intervals = time_intervals[(time_intervals >= Feature.TIME_MIN) & (time_intervals <= Feature.TIME_MAX)]
                percentages.append(np.sum(time_intervals) / maps_duration[video_id])

            if len(percentages) == 0:
                logging.debug('feature {} is invalid: no video clickstream recorded for ratio_duration {}'.format(self.name, percentages))
                return Feature.INVALID_VALUE

            return np.mean(percentages)

        if self.settings['mode'] == 'ratio_played':

            percentages = []
            for i, (video_id, video_data) in enumerate(self.data.groupby(by='video_id')):
                video_data['prev_event'] = video_data['event_type'].shift(1)
                video_data['prev_video_id'] = video_data['video_id'].shift(1)
                video_data['time_diff'] = video_data['date'].diff().dt.total_seconds()
                video_data = video_data.dropna(subset=['time_diff'])
                video_data = video_data[(video_data['time_diff'] >= Feature.TIME_MIN) & (video_data['time_diff'] <= Feature.TIME_MAX)]
                time_intervals_played = video_data[video_data['prev_event'] == 'Video.Play']['time_diff'].values
                time_intervals_played = time_intervals_played[(time_intervals_played >= Feature.TIME_MIN) & (time_intervals_played <= Feature.TIME_MAX)]
                time_intervals_paused = video_data[video_data['prev_event'] == 'Video.Pause']['time_diff'].values
                time_intervals_paused = time_intervals_paused[(time_intervals_paused >= Feature.TIME_MIN) & (time_intervals_paused <= Feature.TIME_MAX)]
                if np.sum(time_intervals_played) != 0:
                    percentages.append(np.sum(time_intervals_paused) / np.sum(time_intervals_played))

            if len(percentages) == 0:
                logging.debug('feature {} is invalid: no video clickstream recorded for ratio_played {}'.format(self.name, percentages))
                return Feature.INVALID_VALUE

            return np.mean(percentages)

        if self.settings['mode'] == 'completed' or self.settings['mode'] == 'entirety':

            self.data = self.data.sort_values(by=['video_id', 'timestamp'])
            self.data['prev_event'] = self.data['event_type'].shift(1)
            self.data['prev_video_id'] = self.data['video_id'].shift(1)
            self.data['current_time'].fillna(self.data.old_time, inplace=True)
            self.data['prev_current_time'] = self.data['current_time'].shift(1)
            self.data = self.data.dropna(subset=['video_id', 'current_time'])
            self.data = self.data[(self.data['prev_event'] == 'Video.Play') & (self.data['video_id'] == self.data['prev_video_id'])]

            completion = self.find_completion(self.data, maps_duration)
            if len(completion) == 0:
                logging.debug('feature {} is invalid: no video clickstream recorded for {}, {}'.format(self.name, self.settings['mode'], completion))
                return Feature.INVALID_VALUE

            return np.mean(np.fromiter(completion.values(), dtype=float))

        if self.settings['mode'] == 'spent':

            percentages = []
            for i, (video_id, video_data) in enumerate(self.data.groupby(by='video_id')):
                if video_id not in maps_duration or math.isnan(maps_duration[video_id]):
                    continue
                video_data['prev_event'] = video_data['event_type'].shift(1)
                video_data['prev_video_id'] = video_data['video_id'].shift(1)
                video_data['time_diff'] = video_data['date'].diff().dt.total_seconds()
                video_data = video_data.dropna(subset=['time_diff'])
                video_data = video_data[(video_data['time_diff'] >= Feature.TIME_MIN) & (video_data['time_diff'] <= Feature.TIME_MAX)]
                time_intervals_played = video_data[video_data['prev_event'] == 'Video.Play']['time_diff'].values
                time_intervals_played = time_intervals_played[(time_intervals_played >= Feature.TIME_MIN) & (time_intervals_played <= Feature.TIME_MAX)]
                time_intervals_paused = video_data[video_data['prev_event'] == 'Video.Pause']['time_diff'].values
                time_intervals_paused = time_intervals_paused[(time_intervals_paused >= Feature.TIME_MIN) & (time_intervals_paused <= Feature.TIME_MAX)]
                percentages.append((np.sum(time_intervals_paused) + np.sum(time_intervals_played)) / maps_duration[video_id])

            if len(percentages) == 0:
                logging.debug('feature {} is invalid: no video clickstream recorded for spent {}'.format(self.name, percentages))
                return Feature.INVALID_VALUE

            return np.mean(percentages)

        if self.settings['mode'] == 'seek_time':
            self.data = self.data[self.data.event_type == self.settings['type']]

            if 'backward' in self.settings['phase']:
                self.data = self.data[self.data['new_time'] < self.data['old_time']]
            else:
                self.data = self.data[self.data['new_time'] > self.data['old_time']]

            self.data['seek_length'] = np.abs(self.data['new_time'] - self.data['old_time'])
            self.data = self.data.merge(self.schedule.query('type=="video"')[['id', 'duration']], left_on='video_id', right_on='id')

            if len(self.data.index) == 0:
                logging.debug('feature {} is invalid: no video clickstream recorded for seek_time {}'.format(self.name, self.settings['phase']))
                return Feature.INVALID_VALUE

            self.data['seek_length'] = self.data['seek_length'] / self.data['duration']
            seek_div_duration = [s / d for d, s in zip(self.data.groupby(by='video_id').max()['duration'], self.data.groupby(by='video_id').sum()['seek_length'])]

            return np.mean(seek_div_duration)

        raise NotImplementedError()