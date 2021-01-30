#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractor.feature.feature import Feature

import logging

'''
The attendance rate of a student s on a given week c since the beginning of the course, is the number of videos
that the student played over to the total number of videos up to that period in time of the course schedule.
'''
class AttendanceRate(Feature):

    def __init__(self, data, settings):
        super().__init__('attendance_rate', data, settings)

    def compute(self):

        if len(self.data.index) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        assert 'course' in self.settings and self.settings['course'].has_schedule()
        taught_videos = set(self.schedule[self.schedule['type'] == 'video']['id'].unique())
        learnt_videos = set(self.data['video_id'].unique())

        if len(taught_videos) == 0:
            logging.info('feature {} is invalid'.format(self.name))
            return Feature.INVALID_VALUE

        return len(learnt_videos & taught_videos) / len(taught_videos)
