#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging
import math

from extractor.feature.feature import Feature
from extractor.feature.time import Time

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

    def find_completion(self, maps_duration):
        maps_spent = {}

        for video_id, video_data in self.data.groupby(by='video_id'):
            if video_id not in maps_duration or math.isnan(maps_duration[video_id]):
                continue

            video_completed = np.zeros(int(maps_duration[video_id]))

            for index, row in video_data.iterrows():
                start_current_time = int(min(row['prev_current_time'], row['current_time']))
                end_current_time = int(max(row['prev_current_time'], row['current_time']))
                video_completed[start_current_time:end_current_time] += 1

            if row.event_type not in ['Video.Pause', 'Video.Stop', 'Video.Load']:
                video_completed[end_current_time:] += 1

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
            logging.debug('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        maps_duration = {v: d for v, d in zip(self.schedule['id'].values, self.schedule['duration'].values)}

        if 'mode' not in self.settings or 'mode' in self.settings and self.settings['mode'] == 'played':
            settings = {**self.settings, **{'ffunc': np.sum}}
            if 'mode' in self.settings and self.settings['mode'] == 'played':
                settings['type'] = 'video'

            maps_spent = {}
            for video_id in self.data.video_id.unique():
                maps_spent[video_id] = Time(self.data, settings).compute()

            percentages = np.zeros(len(maps_spent))
            for i, (video_id, time_spent) in enumerate(maps_spent.items()):
                if video_id in maps_duration:
                    percentages[i] = time_spent / maps_duration[video_id]

            if len(percentages) == 0:
                logging.debug('feature {} is invalid'.format(self.name))
                return Feature.INVALID_VALUE

            return np.mean(percentages)

        if self.settings['mode'] == 'time':
            self.data = self.data[self.data.event_type == self.settings["type"]]
            if 'backward' in self.settings['phase']:
                self.data = self.data[self.data['new_time'] < self.data['old_time']]
            else:
                self.data = self.data[self.data['new_time'] > self.data['old_time']]
            self.data['seek_length'] = np.abs(self.data['new_time'] - self.data['old_time'])
            self.data = self.data.merge(self.schedule.query('type=="video"')[['id', 'duration']], left_on='video_id', right_on='id')
            self.data['seek_length'] = self.data.seek_length / self.data.duration
            maps_spent = {v: d for v, d in zip(self.data.groupby(by='video_id').sum().index,
                                               self.data.groupby(by='video_id').sum()['seek_length'])}
            if len(maps_spent) == 0:
                logging.debug('feature {} is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            return np.mean(np.fromiter(maps_spent.values(), dtype=float))

        self.data = self.data.sort_values(by=['video_id', 'timestamp'])
        self.data['current_time'].fillna(self.data.old_time, inplace=True)
        self.data['prev_event'] = self.data['event_type'].shift(1)
        self.data['prev_video_id'] = self.data['video_id'].shift(1)
        self.data['prev_current_time'] = self.data['current_time'].shift(1)
        self.data['time_diff'] = self.data['date'].diff().dt.total_seconds()
        self.data = self.data.dropna(subset=['time_diff'])
        self.data = self.data.dropna(subset=['current_time'])
        self.data = self.data[
            (self.data['time_diff'] >= Feature.TIME_MIN) & (self.data['time_diff'] <= Feature.TIME_MAX)]

        if self.settings['mode'] in ['completed', 'entirety']:
            self.data = self.data[(self.data['prev_event'].str.contains(self.settings['type'])) & (
                        self.data['video_id'] == self.data['prev_video_id'])]
            completion = self.find_completion(maps_duration)
            if len(completion) == 0:
                logging.debug('feature {} is invalid'.format(self.name))
                return Feature.INVALID_VALUE
            return np.mean(np.fromiter(completion.values(), dtype=float))
