#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from extractors.extractor import Extractor
from helpers.feature_extraction import *

import numpy as np
import sys

'''
He, H., Zheng, Q., Dong, B., & Yu, H. (2018). Measuring Student's Utilization of Video Resources and Its Effect on Academic Performance.
In 2018 IEEE 18th International Conference on Advanced Learning Technologies (ICALT) (pp. 196-198). IEEE.
'''


class HeEtAl(Extractor):
    def __init__(self, name='base'):
        """
        @description: Returns the identifier associated with this feature set.
        """
        super().__init__('he_et_al')

    def getNbFeatures(self):
        """
        @description: Returns the number of expected features.
        """
        return 3

    def getUserFeatures(self, udata, wid, year):
        """
        @description: Returns the user features computed from the udata
        """

        features = [
            self.attendanceRate(udata, wid, year) if len(udata) > 1 else 0,
            self.utilizationRate(udata, wid, year) if len(udata) > 1 else 0,
            self.watchingRatio(udata, wid, year) if len(udata) > 1 else 0,
        ]

        if len(features) != self.getNbFeatures():
            raise Exception("getNbFeatures is not up-to-date: {len(features)} != {self.getNbFeatures()}")

        return list(np.nan_to_num(features))

    def attendanceRate(self, udata, wid, year):
        """
        @description: The attendance rate AR of a student till a given week is the number of videos that the student played over to the total number of videos up to that period in time of the course schedule.
        @requirement: VideoID
        """
        course_schedule = get_dated_videos()
        taught_schedule = course_schedule[(course_schedule['Year'] == year) & (course_schedule['Week'] <= wid)]
        learnt_schedule = udata.drop_duplicates(subset=['VideoID'], keep='first')

        if len(taught_schedule) == 0:
            return 0

        return len(set(learnt_schedule['VideoID']) & set(taught_schedule['VideoID'])) / len(taught_schedule)

    def utilizationRate(self, udata, wid, year):
        """
        @description: The utilization rate UR of a student till a given week is the proportion of video play time activity of the student over the sum of video lengths for all videos up to week c.
        @requirement:
        """
        course_schedule = get_dated_videos()
        taught_schedule = course_schedule[(course_schedule['Year'] == year) & (course_schedule['Week'] <= wid)]

        tmpudata = udata[udata.EventType.isin(['Video.Play', 'Video.Pause', 'Video.Stop'])].copy().sort_values(by='TimeStamp')
        tmpudata['PrevEvent'] = tmpudata['EventType'].shift(1)
        tmpudata['PrevVideoID'] = tmpudata['VideoID'].shift(1)
        tmpudata['TimeDiff'] = tmpudata.TimeStamp.diff()
        tmpudata = tmpudata[(tmpudata['PrevEvent'] == 'Video.Play') & (tmpudata['VideoID'] == tmpudata['PrevVideoID'])]

        if np.sum(taught_schedule['Duration']) == 0:
            return 0

        return np.sum(tmpudata['TimeDiff']) / np.sum(taught_schedule['Duration'])

    def watchingRatio(self, udata, wid, year):
        """
        @description: The student’s overall specialty WR is defined as the ratio between utilization rate and attendance rate. For instance, 1 means that the student s completely watches the video since he/she opened it.
        @requirement: VideoID
        """
        return self.utilizationRate(udata, wid, year) / (self.attendanceRate(udata, wid, year) + sys.float_info.epsilon)